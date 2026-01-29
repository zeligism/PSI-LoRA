import math
import torch
import torch.nn as nn
from src.utils import init_seed


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, num_channels=1, LinearModule=nn.Linear):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 6 * num_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6 * num_channels, 16 * num_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(1),
        )
        self.classifier = nn.Sequential(
            LinearModule(16 * 5 * 5 * num_channels, 120 * num_channels),
            nn.Tanh(),
            LinearModule(120 * num_channels, 84 * num_channels),
            nn.Tanh(),
            LinearModule(84 * num_channels, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_mult=4, p=0.1, LinearModule=nn.Linear):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p)
        self.fc1  = LinearModule(dim, dim * hidden_mult)
        self.fc2  = LinearModule(dim * hidden_mult, dim)

    def forward(self, x):
        h = self.norm(x)
        h = self.activation(self.fc1(h))
        h = self.dropout(self.fc2(h))
        return x + h


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, mlp_dim_token=128, mlp_dim_channel=512, p=0.1, LinearModule=nn.Linear):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            LinearModule(num_tokens, mlp_dim_token),
            nn.GELU(),
            nn.Dropout(p),
            LinearModule(mlp_dim_token, num_tokens),
            nn.Dropout(p),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            LinearModule(hidden_dim, mlp_dim_channel),
            nn.GELU(),
            nn.Dropout(p),
            LinearModule(mlp_dim_channel, hidden_dim),
            nn.Dropout(p),
        )

    def forward(self, x):
        # x: (N, T, D)
        y = self.norm1(x)
        y = self.token_mlp(y.transpose(1, 2)).transpose(1, 2)
        x = x + y
        y = self.norm2(x)
        y = self.channel_mlp(y)
        return x + y


class PatchMLP(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_dim=4,
        num_channels=3,
        hidden_dim=256,
        depth=4,
        num_classes=10,
        hidden_mult=2,
        dropout_p=0.1,
        LinearModule=nn.Linear,
    ):
        super().__init__()
        assert img_size % patch_dim == 0
        self.grid_size = img_size // patch_dim
        num_tokens = self.grid_size * self.grid_size

        self.patchify = nn.Conv2d(
            num_channels, hidden_dim, kernel_size=patch_dim, stride=patch_dim, bias=True
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            ResidualMLPBlock(
                hidden_dim, hidden_mult=hidden_mult, p=dropout_p, LinearModule=LinearModule
            )
            # MixerBlock(
            #     num_tokens=num_tokens,
            #     hidden_dim=hidden_dim,
            #     # mlp_dim_token=int(num_tokens * hidden_mult),
            #     # mlp_dim_channel=int(hidden_dim * hidden_mult),
            #     p=dropout_p,
            #     LinearModule=LinearModule
            # )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = LinearModule(hidden_dim, num_classes)

    def forward(self, x):
        # x: (N, C, H, W)
        x = self.patchify(x)  # (N, hidden_dim, kh, kw)
        x = x.flatten(2).transpose(1, 2)  # (N, T, hidden_dim)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # average tokens -> (N, hidden_dim)
        return self.head(x)  # (N, num_classes)


class LoRALinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int = 0,
        **kwargs
        ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.weight.requires_grad = False

        self.rank = rank
        self.lora_A = nn.Parameter(torch.empty((rank, in_features)))
        self.lora_B = nn.Parameter(torch.empty((out_features, rank)))

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        self.lora_bias = None
        if self.bias is not None:
            self.bias.requires_grad = False
            self.lora_bias = nn.Parameter(torch.zeros((out_features)))

    def forward(self, x: torch.Tensor):
        result = nn.functional.linear(x, self.weight, bias=self.bias)
        result += x @ self.lora_A.T @ self.lora_B.T
        if self.bias is not None:
            result += self.lora_bias.view(1, -1)
        return result

    @property
    def merged_weight(self):
        return self.weight + self.lora_B @ self.lora_A


class SVDLoRALinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int = 0,
        **kwargs
        ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.weight.requires_grad = False

        self.rank = rank
        self.lora_W = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.zeros_(self.lora_W)

        if self.bias is not None:
            self.bias.requires_grad = False
            self.lora_bias = nn.Parameter(torch.zeros((out_features)))

    def forward(self, x: torch.Tensor):
        result = nn.functional.linear(x, self.weight + self.lora_W, bias=self.bias)
        if self.bias is not None:
            result += self.lora_bias.view(1, -1)
        return result

    @torch.no_grad()
    def project(self, W, threshold=None):
        r = self.rank
        U, S, Vt = torch.linalg.svd(W)
        if threshold is not None:
            S.sub_(threshold).clamp_(min=0.0)
            nonzero = (S > 0).sum().item()
            # print(f"  - Projecting to rank {nonzero}/{len(S)} (threshold={threshold:.4f})")
            r = max(nonzero, 1)
        self.lora_W.copy_(U[:, :r] @ torch.diag(S[:r]) @ Vt[:r, :])

    @torch.no_grad()
    def project_back(self, threshold=None):
        self.project(self.lora_W, threshold=threshold)

    @property
    def merged_weight(self):
        return self.weight + self.lora_W


def get_linear_model(in_features, out_features, method, rank, LinearModule=nn.Linear, seed=0):
    # W_0 is basically a kaiming_unform initialized weight
    init_seed(seed)
    full_model = nn.Linear(in_features, out_features, bias=False)
    model = LinearModule(in_features, out_features, bias=False)
    with torch.no_grad():
        if method != "full":
            U, S, Vt = torch.linalg.svd(full_model.weight)
            if hasattr(model, "lora_A") and hasattr(model, "lora_B"):
                model.lora_A.copy_(torch.diag(S[:rank] ** 0.5) @ Vt[:rank, :])
                model.lora_B.copy_(U[:, :rank] @ torch.diag(S[:rank] ** 0.5))
            if hasattr(model, "lora_W"):
                model.lora_W.copy_(U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :])
            # Zero base weights
            nn.init.zeros_(model.weight)
            if model.bias is not None:
                nn.init.zeros_(model.bias)
    
    return model