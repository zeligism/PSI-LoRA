import logging
from collections import defaultdict
import threading

import torch
import torch.nn as nn
import torch.optim as optim

from .utils import (
    _solve,
    safe_inv,
    low_rank_sum,
    scaled_low_rank_sum,
    sqrtm_newton_schulz,
    precond_lora_grad_,
    scalar_quartic_shrinkage_,
    gram_quartic_shrinkage_
)

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG

COMBINED_SUBSPACE_ITERATION = True
RANK_MULTIPLIER = 1.0
USE_MOMENTUM_RESIDUAL = False
INV_FREE = True
LR_LMBD = True
SCALED_OPLORA_LOG_FREQ = 250
SCALED_OPLORA_MOMENTUM_MODE = "normal"  # "normal", "fista", "naive"
OPLORA_MODULE_KEY = "_oplora_key"


def _apply_matrix_metric_power(metric: torch.Tensor, power: float) -> torch.Tensor:
    if power == 1.0:
        return metric
    if power == 0.5:
        return sqrtm_newton_schulz(metric)
    metric_sym = 0.5 * (metric + metric.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(metric_sym)
    eigenvalues = eigenvalues.clamp_min(1e-12).pow(power)
    return (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T


class OPLoraOptimizerBase(optim.Optimizer):
    def __init__(self, model, defaults):

        if isinstance(model, nn.DataParallel):
            logger.warning(
                "Detected torch.nn.DataParallel wrapping. This is slower and can be fragile with hooks. "
                "Prefer DDP via `torchrun` (recommended) or `accelerate launch` to avoid DataParallel. "
                "This optimizer keeps best-effort DataParallel support for robustness."
            )

        self.model = model
        super().__init__(self.model.parameters(), defaults)

        self.lora_modules: dict[str, nn.Module] = {}
        self.other_parameters: dict[str, nn.Parameter] = {}
        # NOTE: when running with `torch.nn.DataParallel`, the model is replicated per device.
        # Hooks execute on replica module objects, which won't match the original module object id.
        # To avoid mismatches at `step()`, we save inputs/grad_outputs by a stable key stored on the module.
        self.inputs: dict[str, torch.Tensor] = {}
        self.grad_outputs: dict[str, torch.Tensor] = {}
        self.is_stale_grad_outputs: defaultdict[str, bool] = defaultdict(lambda: True)
        self._hook_lock = threading.Lock()

        # Handle gradient accumulation internally
        # NOTE: gradient accumulation and compression are experimental and have not yet
        # shown improvements (though lorsum seems to be the least bad compression, yay).
        # The beautiful technique "sketch_centered" is only better than "sketch", unfortunately.
        # I'm keeping the implementation intact for the interested reader.
        self.num_grad_accumulated = 0
        self.accumulated_vectors_limit = 2 ** 14  # to limit memory usage
        self.loss_reduction = "mean"
        self.grad_compression_ratio = 0.5
        self.grad_compression_method = "none"  # "none", "lorsum", "sample_idx", "sketch", "sketch_centered"
        self._cached_batch_size = None

        # Initialize the model and its parameters
        other_parameters_set = set(self.model.parameters())
        for name, module in self.model.named_modules():
            name = "self" + (f".{name}" if name != "" else "")
            if OPLoraOptimizer._is_lora_module(module):
                logger.debug(f"Found LoRA module: {name}")
                # Register hooks to save inputs and gradients
                self.lora_modules[name] = module
                setattr(module, OPLORA_MODULE_KEY, name)  # set global name/path to module
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)
                # Remove from other parameters
                weight_in, weight_out = OPLoraOptimizer.get_lora_weights(module)
                other_parameters_set -= set([weight_in, weight_out])
                self._init_buffers(weight_in, weight_out)  # TODO: remove, init at first step
            else:
                # logger.debug(f"Skipping non-LoRA module: {name}")
                pass

        for name, param in self.model.named_parameters():
            if param in other_parameters_set and param.requires_grad:
                self.other_parameters[name] = param

    def _init_buffers(self, weight_in, weight_out):
        pass

    @staticmethod
    def _is_lora_module(module):
        # HACK: This is a hack to identify LoRA modules. Is there a better way?
        return hasattr(module, 'lora_A') and hasattr(module, 'lora_B')

    @staticmethod
    def get_lora_weights(module):
        assert OPLoraOptimizer._is_lora_module(module), "Module is not a LoRA module"
        weight_in = module.lora_A
        weight_out = module.lora_B
        weight_in = getattr(weight_in, "default", weight_in)
        weight_out = getattr(weight_out, "default", weight_out)
        weight_in = getattr(weight_in, "weight", weight_in)
        weight_out = getattr(weight_out, "weight", weight_out)
        return weight_in, weight_out

    @staticmethod
    def _get_module_key(module: nn.Module) -> str:
        key = getattr(module, OPLORA_MODULE_KEY, None)
        if key is None:
            # Shouldn't happen in normal runs!
            key = f"__unknown__:{module.__class__.__name__}:{id(module)}"
        return key

    def _save_input(self, module, input):
        """
        Save the input to the module for later use in the optimizer step.
        During training we save inputs and mark the grad output as stale.

        Note: the forward pass can actually run under `torch.no_grad()` in some cases,
        (e.g. activation/gradient checkpointing). For our purposes we only need the values
        of the activations, not their gradients, so we check for `training`, not `grad_enabled`.
        """
        if module.training:
            key = self._get_module_key(module)
            if len(input) > 1:
                logger.warning(f"Module {module} received multiple inputs, using the first one.")
            input = input[0]
            input = input.contiguous().view(-1, input.shape[-1]).clone().detach()
            with self._hook_lock:
                if key in self.inputs:
                    # Accumulate inputs if multiple forward passes occur before stepping
                    prev_input = self.inputs[key]
                    input = self._accumulate_vectors(prev_input, input, self.accumulated_vectors_limit)
                self.inputs[key] = input
                self.is_stale_grad_outputs[key] = True  # should save grad_output
        else:
            # Evaluation mode, do nothing
            pass

    def _save_grad_output(self, module, grad_input, grad_output):
        """
        Save the gradient output to the module for later use in the optimizer step.
        grad_output becomes stale when a new input is saved.
        """
        key = self._get_module_key(module)
        with self._hook_lock:
            is_stale = self.is_stale_grad_outputs[key]
        if is_stale:
            if len(grad_output) > 1:
                logger.warning(f"Module {module} received multiple grad outputs, using the first one.")
            grad_output = grad_output[0]
            grad_output = grad_output.view(-1, grad_output.shape[-1]).clone().detach()

            # Cache batch size once
            if self._cached_batch_size is None:
                self._cached_batch_size = grad_output.shape[0]

            with self._hook_lock:
                if key in self.grad_outputs:
                    # Accumulate grad outputs if multiple backward passes occur before stepping
                    prev_grad_output = self.grad_outputs[key]
                    if self.loss_reduction == "mean":
                        prev_grad_output *= prev_grad_output.shape[0]
                    grad_output = self._accumulate_vectors(prev_grad_output, grad_output, self.accumulated_vectors_limit)
                    if self.loss_reduction == "mean":
                        grad_output *= 1.0 / grad_output.shape[0]
                    # Adjust grad_output to account for accumulation
                    self.num_grad_accumulated += 1
                else:
                    self.num_grad_accumulated = 1
                self.grad_outputs[key] = grad_output
                self.is_stale_grad_outputs[key] = False  # saved grad_output

            # Compress vectors after accumulation
            if self.num_grad_accumulated > 1:
                with self._hook_lock:
                    self.inputs[key], self.grad_outputs[key] = (
                        self._compress_grad_factors(
                            self.inputs[key],
                            self.grad_outputs[key],
                            # Assuming a budget of batch_size TODO: What should we use here?
                            ratio=self._cached_batch_size,
                            # ratio=self.grad_compression_ratio,
                            lorsum_args={
                                "num_iters": self.defaults["lookahead_iters"],
                                "lmbd": self.defaults["lr"] * self.defaults["lmbd"],
                            },
                            method=self.grad_compression_method,
                        )
                    )


    @staticmethod
    def _accumulate_vectors(prev_vectors, new_vectors, limit=10**10):
        vector = torch.cat([prev_vectors, new_vectors], dim=0)
        if vector.shape[0] > limit:
            logger.warning("Accumulated vectors exceeded max limit, truncating oldest samples.")
            vector = vector[-limit:, :]
        return vector

    @staticmethod
    def _compress_grad_factors(input, grad_output, ratio=0.5, lorsum_args={}, method="none"):
        assert method in [
            "lorsum", "sample_idx", "sketch", "sketch_centered", "none"
        ], f"Unknown compression method: {method}"
        compressed_len = round(ratio) if ratio > 1.0 else max(1, round(ratio * input.shape[0]))
        if method == "lorsum":
            input, grad_output_T = low_rank_sum(
                factors=(
                    (input[:compressed_len, :], grad_output[:compressed_len, :].T),
                    (input[compressed_len:, :], grad_output[compressed_len:, :].T),
                ),
                coefficients=(1.0, 1.0),
                inv_free=INV_FREE,
                **lorsum_args
            )
            return input, grad_output_T.T

        elif method == "sample_idx":
            weights = input.square().sum(dim=1).sqrt() * grad_output.square().sum(dim=1).sqrt()
            sampled_idx = torch.multinomial(weights, num_samples=compressed_len, replacement=False)
            # grad_output = grad_output * (input.shape[0] / compressed_len)  # rescale
            return input[sampled_idx, :], grad_output[sampled_idx, :]

        elif method == "sketch":
            full_len = input.shape[0]
            rademacher = torch.randint(0, 2, (compressed_len, full_len)).to(input) * 2 - 1
            sketch_matrix = rademacher / (compressed_len ** 0.5)  # E[R^T R] = I
            input = sketch_matrix @ input
            grad_output = sketch_matrix @ grad_output
            return input, grad_output

        elif method == "sketch_centered":
            # Get batch means and centered vectors
            full_len = input.shape[0]
            input_mean = input.mean(dim=0, keepdim=True)
            grad_output_mean = grad_output.mean(dim=0, keepdim=True)
            input_centered = input - input_mean
            grad_output_centered = grad_output - grad_output_mean
            # Compute sketch of centered vectors (because less variance)
            rademacher = torch.randint(0, 2, (compressed_len, full_len)).to(input) * 2 - 1
            sketch_matrix = rademacher / (compressed_len ** 0.5)  # E[R^T R] = I
            # Augment centered vectors with means
            # (we can do this because: S_aug^T X_aug ≈ S_c^T X_c + B mu_s mu_x^T = S^T X)
            input_augmented = [
                sketch_matrix @ input_centered,
                input_mean * (full_len ** 0.5)
            ]
            grad_output_augmented = [
                sketch_matrix @ grad_output_centered,
                grad_output_mean * (full_len ** 0.5)
            ]
            return torch.cat(input_augmented), torch.cat(grad_output_augmented)

        elif method == "none":
            return input, grad_output

        else:
            raise ValueError(f"Unknown compression method: {method}")

    @torch.no_grad()
    def clip_grads_(self, max_grad_norm, use_lora_norm=False, dont_clip=False, use_metric=False):
        # XXX: `dont_clip` is an experimental argument to get the clip scale without applying it.
        total_norm_square = 0.0
        lora_total_norm_square = 0.0

        # Use ||S^T X||_F^2 = Tr((S^T X)^T S^T X) = Tr(XX^T SS^T) = <XX^T, SS^T>_F --> O(B^2 max{m,n})
        # Use ||S^T X||_KFAC^2 = Tr(( E[SS^T]^-p S^T X E[XX^T]^-p )^T E[SS^T]^-p S^T X E[XX^T]^-p)
        #   = Tr(X E[XX^T]^-2p X^T S E[SS^T]^-2p S^T)
        for name, module in self.lora_modules.items():
            key = getattr(module, OPLORA_MODULE_KEY, name)
            input = self.inputs.get(key, None)
            grad_output = self.grad_outputs.get(key, None)
            if input is None or grad_output is None:
                continue
            group = self.param_groups[0]
            weight_in, weight_out = OPLoraOptimizer.get_lora_weights(module)
            if use_metric:
                # XXX: experimental
                assert group["metric_type"] == "diagonal"
                metric_in = self.state[weight_in]["metric"]
                metric_out = self.state[weight_out]["metric"]
                metric_power = group["metric_power"]
                metric_in = metric_in.pow(metric_power)
                metric_out = metric_out.pow(metric_power)
                metric_in = metric_in.add(group["damping"])
                metric_out = metric_out.add(group["damping"])
                input_cov = torch.sum(input.square() * metric_in.pow(-1).view(1, -1), dim=1)
                grad_output_cov = torch.sum(grad_output.square() * metric_out.pow(-1).view(1, -1), dim=1)
            else:
                input_cov = input @ input.T
                grad_output_cov = grad_output @ grad_output.T
            total_norm_square += torch.sum(input_cov * grad_output_cov).item()
            lora_total_norm_square += weight_in.grad.square().sum().item() + weight_out.grad.square().sum().item()
            # if group["step"] % SCALED_OPLORA_LOG_FREQ == 0:
            #     logger.info(f"[Clipping] step={group['step']:d} module={name}, input_cov={input_cov.mean():.3e}, grad_output_cov={grad_output_cov.mean():.3e}, norm={torch.sum(input_cov * grad_output_cov).sqrt().item():.3e}, total_norm={total_norm_square ** 0.5:.3e}")
        # Compute norm of other parameters as normal
        for name, param in self.other_parameters.items():
            if param.grad is not None:
                total_norm_square += param.grad.square().sum().item()
                lora_total_norm_square += param.grad.square().sum().item()

        # Calculate clip scale
        total_norm = total_norm_square ** 0.5
        clip_coef = max_grad_norm / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)  # PyTorch does something similar, says it's GPU-friendly

        lora_total_norm = lora_total_norm_square ** 0.5
        lora_clip_coef = max_grad_norm / (lora_total_norm + 1e-6)
        lora_clip_coef = min(lora_clip_coef, 1.0)

        # if group["step"] % SCALED_OPLORA_LOG_FREQ == 0:
        #     logger.info(f"[Clipping] step={group['step']:d}, clip_coef={clip_coef:.3e}, lora_clip_coef={lora_clip_coef:.3e} (total_norm={total_norm:.3e}, lora_total_norm={lora_total_norm:.3e}), max_grad_norm={max_grad_norm:.3e}")

        if use_lora_norm:
            clip_coef = lora_clip_coef

        if dont_clip:
            return clip_coef

        # Implicitly clip FULL grads, not lora grads
        # XXX: Will not work correctly with gradient accumulation across steps
        for name, module in self.lora_modules.items():
            key = getattr(module, OPLORA_MODULE_KEY, name)
            if key in self.inputs and key in self.grad_outputs:
                # TODO: Distribute clipping scale evenly?
                self.inputs[key].mul_(clip_coef)
                # self.grad_outputs[key].mul_(clip_coef ** 0.5)
        # Then clip other grads normally
        for name, param in self.other_parameters.items():
            if param.grad is not None:
                param.grad.mul_(clip_coef)

        return clip_coef

    @torch.no_grad()
    def sgd_step_(self, param, lr, momentum=0.0, dampening=0.0, weight_decay=0.0):
        if param.grad is None:
            return
        g = param.grad
        if momentum > 0.0:
            pstate = self.state[param]
            if "momentum_buffer" not in pstate:
                pstate["momentum_buffer"] = param.grad.clone().detach()
            else:
                pstate["momentum_buffer"].mul_(momentum).add_(param.grad, alpha=1.0 - dampening)
            g = pstate["momentum_buffer"]
        if weight_decay > 0.0:
            param.sub_(param, alpha=lr * weight_decay)
        param.sub_(g, alpha=lr)

    @torch.no_grad()
    def adam_step_(self, param, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if param.grad is None:
            return
        pstate = self.state[param]
        if "exp_avg" not in pstate:
            pstate["exp_avg"] = torch.zeros_like(param)
        if "exp_avg_sq" not in pstate:
            pstate["exp_avg_sq"] = torch.zeros_like(param)
        exp_avg, exp_avg_sq = pstate["exp_avg"], pstate["exp_avg_sq"]
        exp_avg.mul_(betas[0]).add_(param.grad, alpha=1 - betas[0])
        exp_avg_sq.mul_(betas[1]).addcmul_(param.grad, param.grad, value=1 - betas[1])
        denom = exp_avg_sq.sqrt().add_(eps)
        pstate["step"] = pstate.get("step", 0) + 1
        bias_correction1 = 1 - betas[0] ** pstate["step"]
        bias_correction2 = 1 - betas[1] ** pstate["step"]
        step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
        if weight_decay > 0.0:
            param.sub_(param, alpha=lr * weight_decay)
        param.addcdiv_(exp_avg, denom, value=-step_size)


# ================================================================
# ====================== OPLoRA Optimizers ======================
# ================================================================

def copy_update_and_maybe_shrink_(
    weight_in, weight_out, weight_in_t, weight_out_t, metric_in=None, metric_out=None, shrinkage_type="none", shrinkage_kwargs={}
):
    # TODO: this function's design is ugly, need to pretty it up
    metrics = None
    if metric_in is not None and metric_out is not None:
        metrics = (metric_in, metric_out)
    if shrinkage_type == "none":
        weight_in.copy_(weight_in_t)
        weight_out.copy_(weight_out_t)
        return 1
    elif shrinkage_type == "scalar":
        return scalar_quartic_shrinkage_(
            (weight_in, weight_out),
            (weight_in_t, weight_out_t),
            metrics=metrics,
            **shrinkage_kwargs,
        )
    elif shrinkage_type == "gram":
        return gram_quartic_shrinkage_(
            (weight_in, weight_out),
            (weight_in_t, weight_out_t),
            metrics=metrics,
            **shrinkage_kwargs,
        )
    else:
        raise ValueError(f"Unknown update type: {shrinkage_type}")


class OPLoraOptimizer(OPLoraOptimizerBase):
    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 0.001,
        lookahead_iters: int = 1,
        lmbd: float = 1e-2,
        lmbd_momentum: float | None = None,  # TODO: remove this?
        momentum: float = 0.0,
        dampening: float = 0.0,  # TODO: might not be necessary
        use_momentum_residual: bool = USE_MOMENTUM_RESIDUAL,
        rank_multiplier: float = RANK_MULTIPLIER,
        start_turn: str = "in",
        max_grad_norm: float = 0.0,
        sgd_args_for_other_parameters: dict = {},
        quartic_kernel: str = "none",  # "none", "scalar_quartic", "gram_quartic"
    ) -> None:
        # sanity checks
        assert lr > 0.0
        assert lookahead_iters >= 0
        assert lmbd >= 0.0

        defaults = dict(
            lr=lr,
            base_lr=lr,
            lookahead_iters=lookahead_iters,
            lmbd=lmbd,
            lmbd_momentum=lmbd if lmbd_momentum is None else lmbd_momentum,
            momentum=momentum,
            dampening=dampening,
            use_momentum_residual=use_momentum_residual,
            rank_multiplier=rank_multiplier,
            start_turn=start_turn,
            max_grad_norm=max_grad_norm,
            quartic_kernel=quartic_kernel,
        )
        super().__init__(model, defaults)
        sgd_args_for_other_parameters.setdefault("lr", lr)
        sgd_args_for_other_parameters.setdefault("momentum", momentum)
        sgd_args_for_other_parameters.setdefault("dampening", dampening)
        sgd_args_for_other_parameters.setdefault("weight_decay", 0.0)
        self.sgd_args_for_other_parameters = sgd_args_for_other_parameters

    def _init_buffers(self, weight_in, weight_out):
        pstate_in = self.state[weight_in]
        pstate_out = self.state[weight_out]

        if "momentum_factor" in pstate_in and "momentum_factor" in pstate_out:
            return

        # Initialize low-rank buffer of full momentum (i.e., momentum of grad_weight)
        momentum_rank = round(weight_in.shape[0] * self.defaults["rank_multiplier"])
        pstate_in["momentum_factor"] = torch.randn(momentum_rank, weight_in.shape[1]).to(weight_in)
        pstate_out["momentum_factor"] = torch.zeros(weight_out.shape[0], momentum_rank).to(weight_out)

        if self.defaults["use_momentum_residual"]:
            res_rank = weight_out.shape[0]
            pstate_in["momentum_residual_factor"] = torch.randn(res_rank, weight_in.shape[1]).to(weight_in)
            pstate_out["momentum_residual_factor"] = torch.zeros(weight_out.shape[0], res_rank).to(weight_out)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        momentum = group["momentum"]
        dampening = group["dampening"]
        max_grad_norm = group["max_grad_norm"]

        # Gradient clipping
        if max_grad_norm > 0.0:
            self.clip_grads_(max_grad_norm)

        # OPLoRA + momentum step
        for name, module in self.lora_modules.items():
            # Get (and clear out) input and grad_output
            key = getattr(module, OPLORA_MODULE_KEY, name)
            input = self.inputs.pop(key, None)
            grad_output = self.grad_outputs.pop(key, None)
            self.is_stale_grad_outputs[key] = True

            weight_in, weight_out = OPLoraOptimizer.get_lora_weights(module)

            if input is None or grad_output is None:
                logger.warning(f"Fallback SGD step on {name} since one of inputs or gradients are missing!")
                self.sgd_step_(weight_in, lr)
                self.sgd_step_(weight_out, lr)
                continue
            
            input, grad_output = input.to(weight_in.dtype), grad_output.to(weight_in.dtype)

            # Step
            if COMBINED_SUBSPACE_ITERATION:
                self.subspace_iteration_step_combined(
                    weight_in, weight_out, input, grad_output, lr, momentum=momentum, dampening=dampening
                )
            else:
                self.subspace_iteration_step(
                    weight_in, weight_out, input, grad_output, lr, momentum=momentum, dampening=dampening
                )

            # Update momentum
            self.update_lowrank_momentum(
                weight_in, weight_out, input, grad_output, lr, momentum=momentum, dampening=dampening
            )

        # SGD + momentum step for other parameters
        for name, param in self.other_parameters.items():
            self.sgd_step_(param, **self.sgd_args_for_other_parameters)

        return loss

    def subspace_iteration_step(self, weight_in, weight_out, input, grad_output, lr, momentum=0.0, dampening=0.0):
        raise DeprecationWarning("subspace_iteration_step is deprecated")
        # Preconditioned low-rank gradient descent
        weight_in_t, weight_out_t = low_rank_sum(
            factors=(
                (weight_in, weight_out),  # weight = weight_out @ weight_in
                (input, grad_output.T),  # grad_weight = grad_output^T @ input
            ),
            coefficients=(1.0, -lr * (1 - dampening)),
            num_iters=self.defaults["lookahead_iters"],
            lmbd=lr * self.defaults["lmbd"],
            inv_free=INV_FREE,
        )
        weight_in.copy_(weight_in_t)
        weight_out.copy_(weight_out_t)

        # Preconditioned low-rank momentum
        if momentum > 0.0:
            buf_factor_in = self.state[weight_in]["momentum_factor"]
            buf_factor_out = self.state[weight_out]["momentum_factor"]
            rank = weight_in.shape[0]
            eye = torch.eye(rank).mul(lr * self.defaults["lmbd"]).to(weight_in_t)
            precond_in = safe_inv(weight_out_t.T @ weight_out_t + eye) @ weight_out_t.T
            precond_out = weight_in_t.T @ safe_inv(weight_in_t @ weight_in_t.T + eye)
            weight_in.sub_(
                (precond_in @ buf_factor_out) @ buf_factor_in,
                alpha=lr * momentum
            )
            weight_out.sub_(
                buf_factor_out @ (buf_factor_in @ precond_out),
                alpha=lr * momentum
            )

    def subspace_iteration_step_combined(self, weight_in, weight_out, input, grad_output, lr, momentum=0.0, dampening=0.0):
        pstate_in = self.state[weight_in]
        pstate_out = self.state[weight_out]
        buf_factor_in = pstate_in["momentum_factor"]
        buf_factor_out = pstate_out["momentum_factor"]

        # M = grad_output.T @ input + momentum * (buf_factor_out @ buf_factor_in)
        # U, Vt = naive_orthogonalize(M)

        # Preconditioned low-rank momentum
        weight_in_t, weight_out_t = low_rank_sum(
            factors=(
                (weight_in, weight_out),  # weight = weight_out @ weight_in
                (input, grad_output.T),  # grad_weight = grad_output^T @ input
                (buf_factor_in, buf_factor_out),  # buf = buf_factor_in @ buf_factor_out
                # (Vt, U),
            ),
            coefficients=(1.0, -lr * (1 - dampening), -lr * momentum),
            # coefficients=(1.0, -lr),
            num_iters=self.defaults["lookahead_iters"],
            lmbd=lr * self.defaults["lmbd"],
            inv_free=INV_FREE,
        )

        copy_update_and_maybe_shrink_(
            weight_in, weight_out, weight_in_t, weight_out_t, self.defaults["quartic_kernel"]
        )

        # # XXX: DEBUG
        # G = grad_output.T @ input
        # M = G.add(buf_factor_out @ buf_factor_in, alpha=momentum)
        # U, S, Vt = torch.linalg.svd((weight_out @ weight_in).sub(M, alpha=lr))
        # r = weight_in.shape[0]
        # weight_in.copy_(torch.diag(S[:r]) @ Vt[:r, :])
        # weight_out.copy_(U[:, :r])

    def update_lowrank_momentum(self, weight_in, weight_out, input, grad_output, lr, momentum, dampening=0.0):
        pstate_in = self.state[weight_in]
        pstate_out = self.state[weight_out]
        buf_factor_in = pstate_in["momentum_factor"]
        buf_factor_out = pstate_out["momentum_factor"]

        if not self.defaults["use_momentum_residual"]:
            # low_rank_momentum <- alpha * low_rank_momentum + grad_weight
            buf_factor_in_next, buf_factor_out_next = low_rank_sum(
                factors=[
                    (buf_factor_in, buf_factor_out),
                    (input, grad_output.T),
                ],
                coefficients=[momentum, 1.0 - dampening],
                num_iters=self.defaults["lookahead_iters"],
                lmbd=lr * self.defaults["lmbd_momentum"],
                inv_free=INV_FREE,
            )
            buf_factor_in.copy_(buf_factor_in_next)
            buf_factor_out.copy_(buf_factor_out_next)

            # # XXX: DEBUG
            # M = (grad_output.T @ input).add(buf_factor_out @ buf_factor_in, alpha=momentum)
            # # U, S, Vt = torch.linalg.svd(M)
            # # r = buf_factor_in.shape[0]
            # # buf_factor_in.copy_(torch.diag(S[:r] ** 0.5) @ Vt[:r, :])
            # # buf_factor_out.copy_(U[:, :r] @ torch.diag(S[:r] ** 0.5))
            # pstate_in["momentum_factor"] = M
            # pstate_out["momentum_factor"] = torch.eye(M.shape[0]).to(M)

        else:
            res_factor_in = pstate_in["momentum_residual_factor"]
            res_factor_out = pstate_out["momentum_residual_factor"]
            # low_rank_momentum <- alpha * low_rank_momentum + grad_weight + residual
            buf_factor_in_next, buf_factor_out_next = low_rank_sum(
                factors=[
                    (buf_factor_in, buf_factor_out),
                    (input, grad_output.T),
                    (res_factor_in, res_factor_out)
                ],
                coefficients=[momentum, 1.0 - dampening, 1.0],
                num_iters=self.defaults["lookahead_iters"],
                lmbd=lr * self.defaults["lmbd_momentum"],
                inv_free=INV_FREE,
            )
            # residual <- residual + alpha * low_rank_momentum + grad_weight - low_rank_momentum_next
            res_factor_in_next, res_factor_out_next = low_rank_sum(
                factors=[
                    (res_factor_in, res_factor_out),
                    (buf_factor_in, buf_factor_out),
                    (input, grad_output.T),
                    (buf_factor_in_next, buf_factor_out_next),
                ],
                coefficients=(1.0, momentum, 1.0, -1.0),
                num_iters=self.defaults["lookahead_iters"],
                lmbd=lr * self.defaults["lmbd_momentum"],
                inv_free=INV_FREE,
            )
            buf_factor_in.copy_(buf_factor_in_next)
            buf_factor_out.copy_(buf_factor_out_next)
            res_factor_in.copy_(res_factor_in_next)
            res_factor_out.copy_(res_factor_out_next)


class OPLoraOptimizerProj(OPLoraOptimizerBase):
    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 0.001,
        lookahead_iters: int = 1,
        lmbd: float = 1e-2,
        lmbd_momentum: float | None = None,
        momentum: float = 0.0,
        dampening: float = 0.0,
        start_turn: str = "in",
        max_grad_norm: float = 0.0,
        sgd_args_for_other_parameters: dict = {},
    ) -> None:
        # sanity checks
        assert lr > 0.0
        assert lookahead_iters >= 0
        assert lmbd >= 0.0
        defaults = dict(
            lr=lr,
            lookahead_iters=lookahead_iters,
            lmbd=lmbd,
            lmbd_momentum=lmbd_momentum or lmbd,
            momentum=momentum,
            dampening=dampening,
            start_turn=start_turn,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(model, defaults)
        sgd_args_for_other_parameters.setdefault("lr", lr)
        sgd_args_for_other_parameters.setdefault("momentum", momentum)
        sgd_args_for_other_parameters.setdefault("dampening", dampening)
        sgd_args_for_other_parameters.setdefault("weight_decay", 0.0)
        self.sgd_args_for_other_parameters = sgd_args_for_other_parameters

    def _init_buffers(self, weight_in, weight_out):
        return

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        momentum = group["momentum"]
        dampening = group["dampening"]
        max_grad_norm = group["max_grad_norm"]

        # Gradient clipping
        if max_grad_norm > 0.0:
            self.clip_grads_(max_grad_norm)

        # OPLoRA + momentum step
        for name, module in self.lora_modules.items():
            key = getattr(module, OPLORA_MODULE_KEY, name)
            input = self.inputs.pop(key, None)
            grad_output = self.grad_outputs.pop(key, None)
            self.is_stale_grad_outputs[key] = True

            weight_in, weight_out = OPLoraOptimizer.get_lora_weights(module)

            if input is None or grad_output is None:
                logger.warning(f"Fallback SGD step on {name} since one of inputs or gradients are missing!")
                self.sgd_step_(weight_in, lr)
                self.sgd_step_(weight_out, lr)
                continue

            input, grad_output = input.to(weight_in.dtype), grad_output.to(weight_in.dtype)

            # Preconditioned low-rank gradient descent
            weight_in_t, weight_out_t = low_rank_sum(
                factors=(
                    (weight_in, weight_out),  # weight = weight_out @ weight_in
                    (input, grad_output.T),  # grad_weight = grad_output^T @ input
                ),
                coefficients=(1.0, -lr * (1 - dampening)),
                num_iters=self.defaults["lookahead_iters"],
                lmbd=lr * self.defaults["lmbd"] if LR_LMBD else self.defaults["lmbd"],
                inv_free=INV_FREE,
            )
            weight_in.copy_(weight_in_t)
            weight_out.copy_(weight_out_t)

            self.proj_momentum_and_apply(
                weight_in, weight_out, input, grad_output, lr, momentum, dampening=dampening
            )

        # SGD + momentum step for other parameters
        for name, param in self.other_parameters.items():
            self.sgd_step_(param, **self.sgd_args_for_other_parameters)

        return loss

    def proj_momentum_and_apply(self, weight_in, weight_out, input, grad_output, lr, momentum, dampening=0.0):
        pstate_in = self.state[weight_in]
        pstate_out = self.state[weight_out]

        # Compute preconditioned gradients
        lmbd = lr * self.defaults["lmbd"] if LR_LMBD else self.defaults["lmbd"]
        eye = torch.eye(weight_in.shape[0]).mul(lmbd).to(weight_in)
        precond_in = _solve(weight_out.T @ weight_out + eye, weight_out.T, inv_free=INV_FREE)
        precond_out = _solve(weight_in @ weight_in.T + eye, weight_in, inv_free=INV_FREE).T
        grad_weight_in = (precond_in @ grad_output.T) @ input
        grad_weight_out = grad_output.T @ (input @ precond_out)

        if "prev" not in pstate_in and "prev" not in pstate_out:
            pstate_in["momentum_buffer"] = grad_weight_in.clone().detach()
            pstate_out["momentum_buffer"] = grad_weight_out.clone().detach()
            pstate_in["prev"] = weight_in.clone().detach()
            pstate_out["prev"] = weight_out.clone().detach()
            return

        # Project momentum
        buf_in, buf_out = pstate_in["momentum_buffer"], pstate_out["momentum_buffer"]
        prev_in, prev_out = pstate_in["prev"], pstate_out["prev"]
        proj_buf_in = (precond_in @ prev_out) @ buf_in
        proj_buf_out = buf_out @ (prev_in @ precond_out)
        buf_in.copy_(
            proj_buf_in.mul(momentum).add_(grad_weight_in, alpha=1.0 - dampening)
        )
        buf_out.copy_(
            proj_buf_out.mul(momentum).add_(grad_weight_out, alpha=1.0 - dampening)
        )
        pstate_in["prev"] = weight_in.clone().detach()
        pstate_out["prev"] = weight_out.clone().detach()

        # Apply
        weight_in.sub_(proj_buf_in, alpha=lr * momentum)
        weight_out.sub_(proj_buf_out, alpha=lr * momentum)


# ================================================================
# =================== Scaled OPLoRA Optimizers ===================
# ================================================================

class ScaledOPLoraOptimizer(OPLoraOptimizerBase):
    METRIC_TYPES = ("dense", "diagonal", "lowrank")

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 0.001,
        lookahead_iters: int = 1,
        lmbd: float = 0.01,
        betas: float = (0.9, 0.999),
        damping: float = 0.1,
        max_grad_norm: float = 0.0,
        metric_type: str = "lowrank",
        init_metric_scale: float = 1.0,
        kfac_metric: bool = True,
        metric_power: float = 1.0,
        loss_reduction: str = "mean",
        rank_multiplier: float = RANK_MULTIPLIER,
        start_precondition_step: int = 0,
        adam_args_for_other_parameters: dict = {},
        quartic_kernel: str = "none",
        quartic_kernel_kwargs: dict = {},
        beta2_warmup: bool = False,
    ) -> None:
        # sanity checks
        assert lr > 0.0
        assert lookahead_iters >= 0
        assert lmbd >= 0.0
        assert metric_type in self.METRIC_TYPES
        assert 0.0 <= metric_power <= 1.0
        defaults = dict(
            lr=lr,
            lookahead_iters=lookahead_iters,
            lmbd=lmbd,
            betas=betas,
            damping=damping,
            max_grad_norm=max_grad_norm,
            metric_type=metric_type,
            init_metric_scale=init_metric_scale,
            kfac_metric=kfac_metric,
            metric_power=metric_power,
            loss_reduction=loss_reduction,
            rank_multiplier=rank_multiplier,
            start_precondition_step=start_precondition_step,
            quartic_kernel=quartic_kernel,
            quartic_kernel_kwargs=quartic_kernel_kwargs,
            beta2_warmup=beta2_warmup,
        )
        super().__init__(model, defaults)
        # adam's scale is different, don't rely on the same lr for kfac scale
        adam_args_for_other_parameters.setdefault("lr", 1e-3)
        adam_args_for_other_parameters.setdefault("betas", (0.9, 0.999))
        adam_args_for_other_parameters.setdefault("eps", 1e-8)
        adam_args_for_other_parameters.setdefault("weight_decay", 0.0)
        self.adam_args_for_other_parameters = adam_args_for_other_parameters

    def _init_buffers(self, weight_in, weight_out):
        pstate_in = self.state[weight_in]
        pstate_out = self.state[weight_out]
        group = self.param_groups[0]

        if "metric" in pstate_in and "metric" in pstate_out:
            return

        init_scale = group["init_metric_scale"]
        if group["metric_type"] == "dense":
            pstate_in["metric"] = torch.eye(weight_in.shape[1]).to(weight_in).mul(init_scale)
            pstate_out["metric"] = torch.eye(weight_out.shape[0]).to(weight_out).mul(init_scale)
        elif group["metric_type"] == "diagonal":
            pstate_in["metric"] = torch.ones(weight_in.shape[1]).to(weight_in).mul(init_scale)
            pstate_out["metric"] = torch.ones(weight_out.shape[0]).to(weight_out).mul(init_scale)
        elif group["metric_type"] == "lowrank":
            metric_rank = min(1, round(weight_in.shape[0] * self.defaults["rank_multiplier"]))
            if True:
                pstate_in["metric_A"] = torch.randn(metric_rank, weight_in.shape[1]).to(weight_in)
                pstate_in["metric_B"] = pstate_in["metric_A"].T.clone().detach().mul(init_scale / metric_rank)
                pstate_out["metric_A"] = torch.randn(metric_rank, weight_out.shape[0]).to(weight_out)
                pstate_out["metric_B"] = pstate_out["metric_A"].T.clone().detach().mul(init_scale / metric_rank)
            else:
                # TODO: remove
                pstate_in["metric_A"] = torch.randn(metric_rank, weight_in.shape[1]).to(weight_in)
                Q, _ = torch.qr(pstate_in["metric_A"].T)
                pstate_in["metric_B"] = Q.mul(init_scale / metric_rank)
                pstate_out["metric_A"] = torch.randn(metric_rank, weight_out.shape[0]).to(weight_out)
                Q, _ = torch.qr(pstate_out["metric_A"].T)
                pstate_out["metric_B"] = Q.mul(init_scale / metric_rank)
        else:
            raise ValueError(f"Unknown metric type: {group['metric_type']}")

        momentum_rank = round(weight_in.shape[0] * self.defaults["rank_multiplier"])
        pstate_in["momentum_factor"] = torch.randn(momentum_rank, weight_in.shape[1]).to(weight_in)
        pstate_out["momentum_factor"] = torch.zeros(weight_out.shape[0], momentum_rank).to(weight_out)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        group["step"] = group.get("step", 0) + 1
        max_grad_norm = group["max_grad_norm"]

        # TODO: removing debug info later

        # Gradient clipping
        clip_grads_inplace = True  # TODO: hardcode True, clipping grads directly seems to be superior
        if max_grad_norm > 0.0:
            clip_coef = self.clip_grads_(max_grad_norm, dont_clip=not clip_grads_inplace, use_metric=False, use_lora_norm=False)
            # XXX: experimental
            # self.defaults["quartic_kernel"] = "scalar"
            # self.defaults["quartic_kernel_kwargs"] = {
            #     "sigma": 0.001,
            #     "alpha": 0.0002,
            # }

        # OPLoRA
        for name, module in self.lora_modules.items():
            key = getattr(module, OPLORA_MODULE_KEY, name)
            input = self.inputs.pop(key, None)
            grad_output = self.grad_outputs.pop(key, None)
            self.is_stale_grad_outputs[key] = True

            weight_in, weight_out = OPLoraOptimizer.get_lora_weights(module)

            if input is None or grad_output is None:
                if weight_in.grad is None or weight_out.grad is None:
                    continue
                logger.warning(f"Fallback SGD step on {name} since one of inputs or gradients are missing!")
                self.sgd_step_(weight_in, lr)
                self.sgd_step_(weight_out, lr)
                continue

            input, grad_output = input.to(weight_in.dtype), grad_output.to(weight_in.dtype)

            if not clip_grads_inplace:
                lr = group["lr"] * clip_coef

            metric_in, metric_out = self.compute_metrics_and_update(
                weight_in, weight_out, input, grad_output
            )
            if isinstance(metric_in, tuple):
                metric_in, metric_in_inv = metric_in
                metric_out, metric_out_inv = metric_out
                precomputed_metric_inverses = (metric_in_inv, metric_out_inv)
            else:
                precomputed_metric_inverses = None

            # TODO: remove, not used
            if group["step"] < group["start_precondition_step"]:
                self.adam_step_(weight_in, lr, betas=(beta1, beta2))
                self.adam_step_(weight_out, lr, betas=(beta1, beta2))
                if group["step"] == group["start_precondition_step"] - 1:
                    pass  # TODO: delete stats?
                continue

            # How to compute lmbd? Make it scaled with lr
            lmbd = group["lr"] * self.defaults["lmbd"] if LR_LMBD else self.defaults["lmbd"]
            lmbd = max(lmbd, 1e-5)  # don't make it go too small

            if SCALED_OPLORA_MOMENTUM_MODE == "fista":
                pstate_in = self.state[weight_in]
                pstate_out = self.state[weight_out]
                if "prev" not in pstate_in and "prev" not in pstate_out:
                    pstate_in["prev"] = weight_in.clone().detach()
                    pstate_out["prev"] = weight_out.clone().detach()

                weight_in_proj, weight_out_proj = scaled_low_rank_sum(
                    factors=[
                        (weight_in, weight_out),  # weight = weight_out @ weight_in
                        (input, grad_output.T),  # grad_weight = grad_output^T @ input
                    ],
                    coefficients=[1.0, -lr],
                    metrics=(metric_in, metric_out),
                    num_iters=self.defaults["lookahead_iters"],
                    lmbd=lmbd,
                    inv_free=INV_FREE,
                )

                weight_in_heavyball, weight_out_heavyball = low_rank_sum(
                    factors=(
                        (weight_in_proj, weight_out_proj),
                        (pstate_in["prev"], pstate_out["prev"]),
                    ),
                    coefficients=(1.0 + beta1, -beta1),
                    num_iters=self.defaults["lookahead_iters"],
                    lmbd=lmbd,
                    inv_free=INV_FREE,
                )

                pstate_in["prev"] = weight_in_proj.clone().detach()
                pstate_out["prev"] = weight_out_proj.clone().detach()
                weight_in.copy_(weight_in_heavyball)
                weight_out.copy_(weight_out_heavyball)
            elif SCALED_OPLORA_MOMENTUM_MODE == "naive":
                if "momentum_buffer" not in self.state[weight_in]:
                    self.state[weight_in]["momentum_buffer"] = torch.zeros_like(weight_in)
                    self.state[weight_out]["momentum_buffer"] = torch.zeros_like(weight_out)
                buf_in = self.state[weight_in]["momentum_buffer"]
                buf_out = self.state[weight_out]["momentum_buffer"]
                weight_in_proj, weight_out_proj, grad_in, grad_out = scaled_low_rank_sum(
                    factors=[
                        (weight_in, weight_out),  # weight = weight_out @ weight_in
                        (input, grad_output.T),  # grad_weight = grad_output^T @ input
                    ],
                    coefficients=[1.0, -lr],
                    # metrics=(metric_in, metric_out),
                    metrics=(torch.ones_like(metric_in), torch.ones_like(metric_out)),
                    num_iters=self.defaults["lookahead_iters"],
                    lmbd=lmbd,
                    inv_free=INV_FREE,
                    alternating=True,
                    return_grads=True,
                )

                aaa = 0.0
                buf_in.mul_(beta1).add_(grad_in)
                buf_out.mul_(beta1).add_(grad_out)
                buf_in.mul_(beta1).add_(grad_in + aaa * weight_in.sub(weight_in_proj))
                buf_out.mul_(beta1).add_(grad_out + aaa * weight_out.sub(weight_out_proj))
                # weight_in.sub_(weight_in.sub(weight_in_proj), alpha=aaa)
                # weight_out.sub_(weight_out.sub(weight_out_proj), alpha=aaa)
                weight_in.sub_(buf_in, alpha=lr)
                weight_out.sub_(buf_out, alpha=lr)

                # Update low-rank momentum, if any
                # if beta1 > 0.0:
                #     self.update_lowrank_exp_avg(weight_in, weight_out, input, grad_output)
            else:
                # Scaled OPLoRA step
                buf_factor_in = self.state[weight_in]["momentum_factor"]
                buf_factor_out = self.state[weight_out]["momentum_factor"]
                weight_in_t, weight_out_t = scaled_low_rank_sum(
                    factors=[
                        (weight_in, weight_out),  # weight = weight_out @ weight_in
                        (input, grad_output.T),  # grad_weight = grad_output^T @ input
                        (buf_factor_in, buf_factor_out),
                    ],
                    coefficients=[1.0, -lr * (1.0 - beta1), -lr * beta1],
                    metrics=(metric_in, metric_out),
                    # precomputed_metric_inverses=precomputed_metric_inverses,
                    num_iters=self.defaults["lookahead_iters"],
                    lmbd=lmbd,
                    inv_free=INV_FREE,
                )

                shrink = copy_update_and_maybe_shrink_(
                    weight_in, weight_out, weight_in_t, weight_out_t,
                    metric_in=metric_in, metric_out=metric_out,
                    shrinkage_type=self.defaults["quartic_kernel"],
                    shrinkage_kwargs=self.defaults.get("quartic_kernel_kwargs", {}),
                )

                # Update low-rank momentum, if any
                if beta1 > 0.0:
                    self.update_lowrank_exp_avg(weight_in, weight_out, input, grad_output)

            # if group["step"] % SCALED_OPLORA_LOG_FREQ == 0:
            #     if metric_in.ndim == 2:
            #         sv_in = torch.linalg.svdvals(metric_in)
            #         sv_out = torch.linalg.svdvals(metric_out)
            #         mean_in = torch.trace(metric_in).div(len(metric_in))
            #         mean_out = torch.trace(metric_out).div(len(metric_out))
            #     else:
            #         sv_in = metric_in
            #         sv_out = metric_out
            #         mean_in = metric_in.mean()
            #         mean_out = metric_out.mean()
            #     cond_in = (sv_in[0] / sv_in[-1].clamp_min(1e-12)).item() if sv_in.numel() > 0 else float("inf")
            #     cond_out = (sv_out[0] / sv_out[-1].clamp_min(1e-12)).item() if sv_out.numel() > 0 else float("inf")
            #     logger.info(
            #         f"[ScaledOPLoRA] step={group['step']:d} module={name} "
            #         f"mean_in={mean_in.item():.3e} "
            #         f"mean_out={mean_out.item():.3e} "
            #         f"cond_in={cond_in:.3e} cond_out={cond_out:.3e}",
            #     )

        for name, param in self.other_parameters.items():
            adam_args_for_other_parameters = self.adam_args_for_other_parameters.copy()
            lr = adam_args_for_other_parameters.pop("lr", 1e-3)
            lr = lr * clip_coef if not clip_grads_inplace else lr
            sched =  group["lr"] / self.defaults["lr"]
            self.adam_step_(param, lr=sched * lr, **adam_args_for_other_parameters)

        return loss


    def compute_metrics_and_update(
            self,
            weight_in: torch.Tensor,
            weight_out: torch.Tensor,
            input: torch.Tensor,
            grad_output: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:

        group = self.param_groups[0]
        _, beta2 = group["betas"]
        if group["beta2_warmup"]:
            beta2 = min(beta2, 1 - 1 / (group["step"] + 1))
        batch_size = input.shape[0]
        avg_factor = 1.0 / batch_size if group["loss_reduction"] == "sum" else batch_size

        if group["metric_type"] == "dense":
            metric_in = self.state[weight_in]["metric"]
            metric_out = self.state[weight_out]["metric"]

            # Compute EMA of K-FAC or Shampoo preconditioners
            if group["kfac_metric"]:
                input_avg_factor = 1.0 / batch_size
                GtG = input.T @ input
                GGt = grad_output.T @ grad_output
            else:
                input_avg_factor = avg_factor
                GtG = input.T @ (grad_output @ grad_output.T) @ input
                GGt = grad_output.T @ (input @ input.T) @ grad_output
            metric_in.mul_(beta2).add_(GtG, alpha=(1.0 - beta2) * input_avg_factor)
            metric_out.mul_(beta2).add_(GGt, alpha=(1.0 - beta2) * avg_factor)

            # Adding a damping term to the metrics is important
            eye_in = torch.eye(metric_in.shape[0]).to(metric_in)
            eye_out = torch.eye(metric_out.shape[0]).to(metric_out)
            metric_in = metric_in.add(eye_in, alpha=group["damping"])
            metric_out = metric_out.add(eye_out, alpha=group["damping"])
            metric_power = group["metric_power"]
            metric_in = _apply_matrix_metric_power(metric_in, metric_power)
            metric_out = _apply_matrix_metric_power(metric_out, metric_power)

            return metric_in, metric_out

        elif group["metric_type"] == "diagonal":
            metric_in = self.state[weight_in]["metric"]
            metric_out = self.state[weight_out]["metric"]

            # Diagonal metric: just use the diagonal of the outer product
            if group["kfac_metric"]:
                input_avg_factor = 1.0 / batch_size
                GtG = input.square().sum(0)
                GGt = grad_output.square().sum(0)
            else:
                input_avg_factor = avg_factor
                GtG = (input * ((grad_output @ grad_output.T) @ input)).sum(dim=0)
                GGt = (grad_output * ((input @ input.T) @ grad_output)).sum(dim=0)
            metric_in.mul_(beta2).add_(GtG, alpha=(1.0 - beta2) * input_avg_factor)
            metric_out.mul_(beta2).add_(GGt, alpha=(1.0 - beta2) * avg_factor)
            metric_in = metric_in.add(group["damping"])
            metric_out = metric_out.add(group["damping"])
            metric_power = group["metric_power"]
            metric_in = metric_in.pow(metric_power)
            metric_out = metric_out.pow(metric_power)

            return metric_in, metric_out

        elif group["metric_type"] == "lowrank":
            # Low-rank metric: use a low-rank approximation of the K-FAC metric
            assert group["kfac_metric"], "Low-rank metric only supports K-FAC metric"
            assert group["metric_power"] == 1.0, "Low-rank metric does not support metric_power"

            metric_in_A, metric_in_B = self.state[weight_in]["metric_A"], self.state[weight_in]["metric_B"]
            metric_out_A, metric_out_B = self.state[weight_out]["metric_A"], self.state[weight_out]["metric_B"]

            # XXX: work in progress:
            try:
                metric_in_A_next, metric_in_B_next = low_rank_sum(
                    factors=[
                        (metric_in_A, metric_in_B),
                        (input / (batch_size ** 0.5), input.T / (batch_size ** 0.5)),
                    ],
                    coefficients=[beta2, (1 - beta2)],
                    num_iters=self.defaults["lookahead_iters"],
                    lmbd=group["lr"] * self.defaults["lmbd"] if LR_LMBD else self.defaults["lmbd"],
                    inv_free=INV_FREE,
                )
                metric_out_A_next, metric_out_B_next = low_rank_sum(
                    factors=[
                        (metric_out_A, metric_out_B),
                        (grad_output * avg_factor ** 0.5, grad_output.T * avg_factor ** 0.5),
                    ],
                    coefficients=[beta2, (1 - beta2)],
                    num_iters=self.defaults["lookahead_iters"],
                    lmbd=group["lr"] * self.defaults["lmbd"] if LR_LMBD else self.defaults["lmbd"],
                    inv_free=INV_FREE,
                )

            except:
                print("Error in low-rank metric update:")
                print(metric_in_A.norm(), metric_in_A.trace())
                print(metric_in_B.norm(), metric_in_B.trace())
                print("A", (metric_in_B @ metric_in_A).norm(), (metric_in_B @ metric_in_A).trace())
                print(metric_out_A.norm(), metric_out_A.trace())
                print(metric_out_B.norm(), metric_out_B.trace())
                print("B", (metric_out_B @ metric_out_A).norm(), (metric_out_B @ metric_out_A).trace())
                raise

            metric_in_A.copy_(metric_in_A_next)
            metric_in_B.copy_(metric_in_B_next)
            metric_out_A.copy_(metric_out_A_next)
            metric_out_B.copy_(metric_out_B_next)

            metric_in = metric_in_B @ metric_in_A
            metric_out = metric_out_B @ metric_out_A

            # Adding a damping term to the metrics is important
            eye_in = torch.eye(metric_in.shape[0]).mul(group["damping"]).to(metric_in)
            eye_out = torch.eye(metric_out.shape[0]).mul(group["damping"]).to(metric_out)
            metric_in = metric_in.add(eye_in)
            metric_out = metric_out.add(eye_out)

            # Woodbury -> (U V^T + lmbd I)^-1 = lmbd^-1 (I - U (lmbd I + V^T U)^-1 V^T)
            def metric_inv_from_sqrt(U, Vt: torch.Tensor) -> torch.Tensor:
                inner = Vt @ U
                inner.add_(torch.eye(inner.shape[0]).to(inner).mul(group["damping"]))
                full_inv = torch.eye(U.shape[0]).to(U)
                full_inv.sub_(U @ safe_inv(inner) @ Vt)
                full_inv.mul_(1.0 / group["damping"])
                return full_inv

            metric_in_inv = metric_inv_from_sqrt(metric_in_B, metric_in_A)
            metric_out_inv = metric_inv_from_sqrt(metric_out_B, metric_out_A)

            INVERTED = False  # XXX
            if INVERTED:
                return (metric_in, metric_in_inv), (metric_out, metric_out_inv)
            else:
                return metric_in, metric_out

        else:
            raise ValueError(f"Unknown metric type: {group['metric_type']}")

    def update_lowrank_exp_avg(
            self,
            weight_in,
            weight_out,
            input,
            grad_output
        ):

        buf_factor_in = self.state[weight_in]["momentum_factor"]
        buf_factor_out = self.state[weight_out]["momentum_factor"]
        group = self.param_groups[0]
        beta1, _ = group["betas"]

        buf_factor_in_next, buf_factor_out_next = low_rank_sum(
            factors=[
                (buf_factor_in, buf_factor_out),
                (input, grad_output.T),
            ],
            coefficients=[beta1, 1 - beta1],
            num_iters=group["lookahead_iters"],
            lmbd=group["lr"] * self.defaults["lmbd"] if LR_LMBD else self.defaults["lmbd"],
            inv_free=INV_FREE,
        )
        buf_factor_in.copy_(buf_factor_in_next)
        buf_factor_out.copy_(buf_factor_out_next)


# ================================================================
# ================ Preconditioned LoRA Optimizers ================
# ================================================================
class SGDWithPreconditionedLora(OPLoraOptimizerBase):
    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 0.001,
        lmbd: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.0,
    ) -> None:

        defaults = dict(
            lr=lr,
            lmbd=lmbd,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(model, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        momentum = group["momentum"]
        dampening = group["dampening"]
        weight_decay = group["weight_decay"]
        max_grad_norm = group["max_grad_norm"]

        # XXX: experimental (gradnorm based on full model or lora model?)
        # # Gradient clipping
        # if max_grad_norm > 0.0:
        #     self.clip_grads_(max_grad_norm)

        # OPLoRA + momentum step
        for name, module in self.lora_modules.items():
            # Clear gradients
            key = getattr(module, OPLORA_MODULE_KEY, name)
            self.inputs.pop(key, None)
            self.grad_outputs.pop(key, None)
            self.is_stale_grad_outputs[key] = True

            # Precondition then step
            weight_in, weight_out = OPLoraOptimizer.get_lora_weights(module)
            precond_lora_grad_(weight_in, weight_out, lmbd=group["lmbd"], inv_free=INV_FREE)
            self.sgd_step_(weight_in, lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
            self.sgd_step_(weight_out, lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)

        # SGD + momentum step for other parameters
        for name, param in self.other_parameters.items():
            self.sgd_step_(param, lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)

        return loss


class AdamWWithPreconditionedLora(OPLoraOptimizerBase):
    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 0.001,
        lmbd: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-4,
    ) -> None:

        defaults = dict(
            lr=lr,
            lmbd=lmbd,
            betas=betas,
            weight_decay=weight_decay,
        )
        super().__init__(model, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        betas = group["betas"]
        weight_decay = group["weight_decay"]

        # OPLoRA + momentum step
        for name, module in self.lora_modules.items():
            weight_in, weight_out = OPLoraOptimizer.get_lora_weights(module)
            precond_lora_grad_(weight_in, weight_out, lmbd=group["lmbd"], inv_free=INV_FREE)
            self.adam_step_(weight_in, lr, betas=betas, weight_decay=weight_decay)
            self.adam_step_(weight_out, lr, betas=betas, weight_decay=weight_decay)

        # SGD + momentum step for other parameters
        for name, param in self.other_parameters.items():
            self.adam_step_(param, lr, betas=betas, weight_decay=weight_decay)

        return loss


# ============================================================================
# ====================== Experimental OPLoRA Optimizers ======================
# ============================================================================

class OPLoraOptimizerFISTA(OPLoraOptimizerBase):
    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 0.001,
        lookahead_iters: int = 1,
        lmbd: float = 1e-2,
        lmbd_momentum: float | None = None,  # TODO: remove this?
        momentum: float = 0.0,
        dampening: float = 0.0,  # TODO: might not be necessary
        use_momentum_residual: bool = USE_MOMENTUM_RESIDUAL,
        rank_multiplier: float = RANK_MULTIPLIER,
        start_turn: str = "in",
        max_grad_norm: float = 0.0,
        sgd_args_for_other_parameters: dict = {},
        quartic_kernel: str = "scalar",
    ) -> None:
        # sanity checks
        assert lr > 0.0
        assert lookahead_iters >= 0
        assert lmbd >= 0.0

        defaults = dict(
            lr=lr,
            lookahead_iters=lookahead_iters,
            lmbd=lmbd,
            momentum=momentum,
            start_turn=start_turn,
            max_grad_norm=max_grad_norm,
            quartic_kernel=quartic_kernel,
        )
        super().__init__(model, defaults)
        sgd_args_for_other_parameters.setdefault("lr", lr)
        sgd_args_for_other_parameters.setdefault("momentum", momentum)
        sgd_args_for_other_parameters.setdefault("dampening", dampening)
        sgd_args_for_other_parameters.setdefault("weight_decay", 0.0)
        self.sgd_args_for_other_parameters = sgd_args_for_other_parameters

    def _init_buffers(self, weight_in, weight_out):
        pass

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        momentum = group["momentum"]
        max_grad_norm = group["max_grad_norm"]

        # Gradient clipping
        if max_grad_norm > 0.0:
            self.clip_grads_(max_grad_norm)

        # OPLoRA + momentum step
        for name, module in self.lora_modules.items():
            key = getattr(module, OPLORA_MODULE_KEY, name)
            input = self.inputs.pop(key, None)
            grad_output = self.grad_outputs.pop(key, None)
            self.is_stale_grad_outputs[key] = True

            weight_in, weight_out = OPLoraOptimizer.get_lora_weights(module)

            if input is None or grad_output is None:
                logger.warning(f"Fallback SGD step on {name} since one of inputs or gradients are missing!")
                self.sgd_step_(weight_in, lr)
                self.sgd_step_(weight_out, lr)
                continue
            
            input, grad_output = input.to(weight_in.dtype), grad_output.to(weight_in.dtype)

            # Step
            self.subspace_iteration_step(
                weight_in, weight_out, input, grad_output, lr, momentum=momentum,
            )

        # SGD + momentum step for other parameters
        for name, param in self.other_parameters.items():
            self.sgd_step_(param, **self.sgd_args_for_other_parameters)

        return loss

    def subspace_iteration_step(self, weight_in, weight_out, input, grad_output, lr, momentum=0.0):
        pstate_in = self.state[weight_in]
        pstate_out = self.state[weight_out]

        lorsum_args = {
            "num_iters": self.defaults["lookahead_iters"],
            "lmbd": lr * self.defaults["lmbd"] if LR_LMBD else self.defaults["lmbd"],
            "inv_free": INV_FREE,
        }

        if "prev" not in pstate_in and "prev" not in pstate_out:
            pstate_in["prev"] = weight_in.clone().detach()
            pstate_out["prev"] = weight_out.clone().detach()

        weight_in_proj, weight_out_proj = low_rank_sum(
            factors=(
                (weight_in, weight_out),  # weight = weight_out @ weight_in
                (input, grad_output.T),  # grad_weight = grad_output^T @ input
            ),
            coefficients=(1.0, -lr),
            **lorsum_args
        )
        # copy_update_and_maybe_shrink_(
        #     weight_in, weight_out, weight_in_proj, weight_out_proj, self.defaults["quartic_kernel"]
        # )
        # weight_in_proj, weight_out_proj = weight_in, weight_out
        weight_in_heavyball, weight_out_heavyball = low_rank_sum(
            factors=(
                (weight_in_proj, weight_out_proj),
                (pstate_in["prev"], pstate_out["prev"]),
            ),
            coefficients=(1.0 + momentum, -momentum),
            **lorsum_args
        )

        pstate_in["prev"] = weight_in_proj.clone().detach()
        pstate_out["prev"] = weight_out_proj.clone().detach()
        weight_in.copy_(weight_in_heavyball)
        weight_out.copy_(weight_out_heavyball)


class OPLoraOptimizerVR(OPLoraOptimizerBase):
    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 0.001,
        lookahead_iters: int = 1,
        lmbd: float = 1e-2,
        lmbd_momentum: float | None = None,  # TODO: remove this?
        momentum: float = 0.0,
        dampening: float = 0.0,  # TODO: might not be necessary
        use_momentum_residual: bool = USE_MOMENTUM_RESIDUAL,
        rank_multiplier: float = RANK_MULTIPLIER,
        start_turn: str = "in",
        max_grad_norm: float = 0.0,
        sgd_args_for_other_parameters: dict = {},
        which_vr: str = "mars",  # "storm" or "mars"
        correction_strength: float = 0.5,
    ) -> None:
        # sanity checks
        assert lr > 0.0
        assert lookahead_iters >= 0
        assert lmbd >= 0.0
        assert rank_multiplier == 1.0, "OPLoraOptimizerVR only supports rank_multiplier=1.0 for now."

        defaults = dict(
            lr=lr,
            lookahead_iters=lookahead_iters,
            lmbd=lmbd,
            momentum=momentum,
            start_turn=start_turn,
            max_grad_norm=max_grad_norm,
            which_vr=which_vr,
            correction_strength=correction_strength,
        )
        super().__init__(model, defaults)
        sgd_args_for_other_parameters.setdefault("lr", lr)
        sgd_args_for_other_parameters.setdefault("momentum", momentum)
        sgd_args_for_other_parameters.setdefault("dampening", dampening)
        sgd_args_for_other_parameters.setdefault("weight_decay", 0.0)
        self.sgd_args_for_other_parameters = sgd_args_for_other_parameters

    def _init_buffers(self, weight_in, weight_out):
        pass

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "OPLoraOptimizerVR requires a closure."
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group["lr"]
        momentum = group["momentum"]
        corr_strength = group["correction_strength"]
        max_grad_norm = group["max_grad_norm"]
        lorsum_args = {
            "num_iters": self.defaults["lookahead_iters"],
            "lmbd": lr * self.defaults["lmbd"] if LR_LMBD else self.defaults["lmbd"],
            "inv_free": INV_FREE,
        }

        # Evaluate grad at current params
        loss = closure()
        if max_grad_norm > 0.0:
            self.clip_grads_(max_grad_norm)

        # Save grad and param, set to prev param
        for name, param in self.other_parameters.items():
            pstate = self.state[param]
            if param.grad is None:
                continue
            if "prev" not in pstate:
                pstate["prev"] = param.clone()
                pstate["momentum_buffer"] = torch.zeros_like(param).to(param)
            pstate["_cached_grad"] = param.grad.clone()
            prev_param = pstate["prev"].clone()
            pstate["prev"] = param.clone()
            param.copy_(prev_param)

        # For LoRA modules
        for name, module in self.lora_modules.items():
            key = getattr(module, OPLORA_MODULE_KEY, name)
            input = self.inputs.pop(key, None)
            grad_output = self.grad_outputs.pop(key, None)
            self.is_stale_grad_outputs[key] = True
            weight_in, weight_out = OPLoraOptimizer.get_lora_weights(module)
            pstate_in, pstate_out = self.state[weight_in], self.state[weight_out]
            if input is None or grad_output is None:
                continue
            input, grad_output = input.to(weight_in.dtype), grad_output.to(weight_in.dtype)
            if "prev" not in pstate_in:
                pstate_in["prev"], pstate_out["prev"] = weight_in.clone(), weight_out.clone()
                pstate_in["momentum_buffer"] = torch.randn_like(weight_in)
                pstate_out["momentum_buffer"] = torch.zeros_like(weight_out)
            pstate_in["_cached_grad"] = (input, grad_output)
            prev_in, prev_out = pstate_in["prev"].clone(), pstate_out["prev"].clone()
            pstate_in["prev"], pstate_out["prev"] = weight_in.clone(), weight_out.clone()
            weight_in.copy_(prev_in), weight_out.copy_(prev_out)

        # Evaluate grad at previous params
        closure()
        if max_grad_norm > 0.0:
            self.clip_grads_(max_grad_norm)

        # VR update
        for name, param in self.other_parameters.items():
            pstate = self.state[param]
            prev_grad = pstate.pop("_cached_grad")
            buf: torch.Tensor = pstate["momentum_buffer"]

            if group["which_vr"] == "storm":
                damp = 1 - momentum
                buf.mul_(damp).add_(param.grad).sub_(prev_grad, alpha=damp)
                param.sub_(buf, alpha=lr)

            elif group["which_vr"] == "mars":
                buf.mul_(momentum).add_(
                    param.grad, alpha=1.0 - momentum + corr_strength * momentum
                ).sub_(
                    prev_grad, alpha=corr_strength * momentum
                )
                param.sub_(buf, alpha=lr)

            else:
                raise ValueError(f"Unknown variance reduction type: {group['which_vr']}")

        for name, module in self.lora_modules.items():
            key = getattr(module, OPLORA_MODULE_KEY, name)
            prev_input = self.inputs.pop(key, None)
            prev_grad_output = self.grad_outputs.pop(key, None)
            self.is_stale_grad_outputs[key] = True
            weight_in, weight_out = OPLoraOptimizer.get_lora_weights(module)
            pstate_in, pstate_out = self.state[weight_in], self.state[weight_out]
            (input, grad_output) = pstate_in.pop("_cached_grad")
            if prev_input is None or prev_grad_output is None:
                logger.warning(f"Missing VR factors for {name}; skipping this module")
                continue
            prev_input, prev_grad_output = prev_input.to(weight_in.dtype), prev_grad_output.to(weight_in.dtype)

            if group["which_vr"] == "storm":
                damp = momentum
                weight_in_next, weight_out_next = low_rank_sum(
                    factors=(
                        (pstate_in["prev"], pstate_out["prev"]),
                        (input, grad_output.T),
                        (prev_input, prev_grad_output.T),
                        (pstate_in["momentum_buffer"], pstate_out["momentum_buffer"]),
                    ),
                    coefficients=(1.0, -lr, lr * damp, -lr * damp),
                    **lorsum_args
                )
                weight_in.copy_(weight_in_next)
                weight_out.copy_(weight_out_next)

                buf_in_next, buf_out_next = low_rank_sum(
                    factors=(
                        (pstate_in["momentum_buffer"], pstate_out["momentum_buffer"]),
                        (input, grad_output.T),
                        (prev_input, prev_grad_output.T),
                    ),
                    coefficients=(damp, 1.0, -damp),
                    **lorsum_args
                )
                pstate_in["momentum_buffer"].copy_(buf_in_next)
                pstate_out["momentum_buffer"].copy_(buf_out_next)

            elif group["which_vr"] == "mars":
                # see below for derivation
                a = momentum
                s = corr_strength
                weight_in_next, weight_out_next = low_rank_sum(
                    factors=(
                        (pstate_in["prev"], pstate_out["prev"]),
                        (pstate_in["momentum_buffer"], pstate_out["momentum_buffer"]),
                        (input, grad_output.T),
                        (prev_input, prev_grad_output.T),
                    ),
                    coefficients=(
                        1.0,
                        -lr * a,
                        -lr * (1.0 - a + s * a),
                        lr * s * a,
                    ),
                    **lorsum_args
                )
                weight_in.copy_(weight_in_next)
                weight_out.copy_(weight_out_next)

                # c <- g + s * a / (1 - a) * (g - g_prev)
                # m <- a * m + (1 - a) * c
                # m <- a * m + (1 - a + s * a) * g - s * a * g_prev
                buf_in_next, buf_out_next = low_rank_sum(
                    factors=(
                        (pstate_in["momentum_buffer"], pstate_out["momentum_buffer"]),
                        (input, grad_output.T),
                        (prev_input, prev_grad_output.T),
                    ),
                    coefficients=(
                        a,
                        1.0 - a + s * a,
                        -s * a
                    ),
                    **lorsum_args
                )
                pstate_in["momentum_buffer"].copy_(buf_in_next)
                pstate_out["momentum_buffer"].copy_(buf_out_next)

            else:
                raise ValueError(f"Unknown variance reduction type: {group['which_vr']}")

        return loss