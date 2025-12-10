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
    def __init__(self, in_dim, hidden=64, out_dim=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, X, mask):
        # X: (B, M, F), mask: (B, M)
        H = self.net(X)                  # (B, M, d)
        pooled = masked_mean(H, mask)    # (B, d)
        return pooled
