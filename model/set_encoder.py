import torch.nn as nn


def masked_mean(x, mask, dim=1):
    """
    x:    (B, M, d)
    mask: (B, M) with True for real rows, False for padding
    returns: (B, d)
    """
    mask_f = mask.float().unsqueeze(-1)  # (B, M, 1)
    num = (x * mask_f).sum(dim=dim)
    den = mask_f.sum(dim=dim).clamp_min(1.0)  # avoid /0 if set is empty
    return num / den


class SetEncoder(nn.Module):
    def __init__(self,
                 in_dim: int, hidden: int = 128, out_dim: int = 128,num_layers: int = 3,
                 dropout: float = 0.1, use_residual: bool = True, ):
        super().__init__()
        self.use_residual = use_residual and (in_dim == hidden)
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.hidden_mlp = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, X, mask):
        # X: (B, M, F), mask: (B, M)
        H_in = X
        H = self.hidden_mlp(X)  # (B, M, hidden)
        if self.use_residual:
            H = H + H_in

        H = self.proj(H)  # (B, M, out_dim)
        pooled = masked_mean(H, mask)  # (B, out_dim)
        return pooled
