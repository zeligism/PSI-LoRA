import torch
from .utils import init_seed


class LinearSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, in_features, out_features, rank, input_type, seed=0):
        self.in_features = in_features
        self.out_features = out_features
        self.input_type = input_type

        init_seed(seed)
        self.W = torch.randn(self.in_features, self.out_features)

        if self.input_type == "eye":
            self.X = torch.eye(self.in_features)
        elif self.input_type == "random":
            X = torch.randn(self.in_features, self.in_features)
            X, _ = torch.linalg.qr(X @ X.T)
            self.X = X
        else:
            raise NotImplementedError(f"Unknown input type: {self.input_type}")

        self.Y = self.X @ self.W

        if self.input_type == "eye":
            U, S, Vt = torch.linalg.svd(self.W)
            self.W_opt = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
            self.fopt = (self.W - self.W_opt).pow(2).sum().item()
            # print(f"Rank-r approximation error (lowest possible) = {self.fopt}")
        elif self.input_type == "random":
            # XXX: not yet sure about this (look up reduced-rank regression)
            _, _, Vt = torch.linalg.svd(self.Y)
            self.W_opt = self.Y @ Vt[:rank, :].T @ Vt[:rank, :]
            self.fopt = (self.W - self.W_opt).pow(2).sum().item()

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]