import torch.nn as nn


def masked_mean(x, mask, dim=1):
    """
    x:    (B, M, d)
    mask: (B, M) with True for real rows, False for padding
    returns: (B, d)
    """
    mask_f = mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)  # (B, M, 1)
    num = (x * mask_f).sum(dim=dim)
    den = mask_f.sum(dim=dim).clamp_min(1.0)  # avoid /0 if set is empty
    return num / den


class SetEncoder(nn.Module):
    def __init__(self,
                 in_dim: int, hidden: int = 128, out_dim: int = 128,num_layers: int = 3,
                 dropout: float = 0.1, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.input_proj = nn.Linear(in_dim, hidden)
        self.input_act = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout)

        # Residual blocks in hidden space
        # Each block: hidden -> hidden, then ReLU + Dropout, with optional skip
        num_blocks = max(num_layers - 1, 0)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_blocks)
            ]
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, X, mask):
        """
        X:    (B, M, F)   F = in_dim
        mask: (B, M)      True = real row, False = padding
        """
        # Project to hidden space
        H = self.input_proj(X)          # (B, M, hidden)
        H = self.input_act(H)
        H = self.input_dropout(H)

        # Residual blocks
        for block in self.blocks:
            if self.use_residual:
                residual = H
                H_block = block(H)
                H = H_block + residual
            else:
                H = block(H)

        # Project to out_dim
        H = self.proj(H)                # (B, M, out_dim)

        # Pool over the set dimension with mask
        pooled = masked_mean(H, mask)   # (B, out_dim)
        return pooled