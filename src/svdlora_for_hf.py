import torch
import torch.nn as nn
from peft.tuners.lora import Linear
from transformers import TrainerCallback


class SVDLoRALinear(Linear, nn.Module):
    """
    LoRA layer implemented using SVD decomposition for huggingface.
    """
    # Lora implemented in a dense layer
    def __init__(self, base_layer, adapter_name: str, **kwargs):
        super().__init__(base_layer, adapter_name, **kwargs)

        # Add lora_W and redo update_layer manually here
        self.lora_W = nn.ModuleDict({})
        self.lora_W[adapter_name] = nn.Linear(self.in_features, self.out_features, bias=self.lora_bias[adapter_name])
        self.next = {}
        self.prev = {}
        nn.init.zeros_(self.lora_W[adapter_name].weight)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(adapter_name)

    def set_adapter(self, adapter_names, **kwargs):
        # Override set_adapter so that lora_A and lora_B are always turned off
        super().set_adapter(adapter_names, **kwargs)
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                raise NotImplementedError("unmerge is not implemented for this layer")
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise NotImplementedError("_mixed_batch_forward is not implemented for this layer")
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_W.keys():
                    continue
                lora_W = self.lora_W[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_W.weight.dtype)
                assert not self.use_dora[active_adapter]

                # if active_adapter in self.next:
                #     with torch.no_grad():
                #         lora_W.weight.copy_(self.next[active_adapter])  # XXX: fista lookahead
                result = result + lora_W(dropout(x)) * scaling

            result = result.to(torch_result_dtype)

        return result

    @torch.no_grad()
    def project_back(self):
        for adapter_name in self.lora_W.keys():
            r = self.r[adapter_name]
            U, S, Vt = torch.linalg.svd(self.lora_W[adapter_name].weight)
            S_r_sqrt = torch.diag(S[:r]).sqrt()
            self.lora_A[adapter_name].weight.copy_(S_r_sqrt @ Vt[:r, :])
            self.lora_B[adapter_name].weight.copy_(U[:, :r] @ S_r_sqrt)
            if self.lora_bias[adapter_name]:
                self.lora_B[adapter_name].bias.copy_(self.lora_W[adapter_name].bias)
            self.lora_W[adapter_name].weight.copy_(self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight)

    @torch.no_grad()
    def fista(self, momentum, reproject_back=False):
        # XXX: work in progress
        for key in self.lora_W.keys():
            lora_W = self.lora_W[key]
            if key in self.prev:
                self.next[key] = lora_W.weight.mul(1.0 + momentum).sub(self.prev[key], alpha=momentum)
                self.prev[key] = lora_W.weight.clone().detach()
                lora_W.weight.copy_(self.next[key])  # XXX
            else:
                self.prev[key] = lora_W.weight.clone().detach()

        if reproject_back:
            self.project_back()


class SVDCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            for module in model.modules():
                if hasattr(module, "project_back"):
                    module.project_back()


class SVDFISTACallback(TrainerCallback):
    def __init__(self, momentum, reproject_back=False):
        self.momentum = momentum
        self.reproject_back = reproject_back

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            for module in model.modules():
                if hasattr(module, "project_back"):
                    module.project_back()
                if hasattr(module, "fista"):
                    module.fista(self.momentum, self.reproject_back)
