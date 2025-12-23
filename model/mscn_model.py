import torch
import torch.nn as nn

from model.set_encoder import SetEncoder


class MSCNModel(nn.Module):
    def __init__(
        self,
        table_dim: int,
        join_dim: int,
        pred_dim: int,
        hidden_set: int = 128,
        hidden_out: int = 256,
        dropout: float = 0.1,
        num_set_layers: int = 3,
    ):
        super().__init__()
        self.table_mlp = SetEncoder(
            in_dim=table_dim,
            hidden=hidden_set,
            out_dim=hidden_set,
            num_layers=num_set_layers,
            dropout=dropout,
        )
        self.join_mlp = SetEncoder(
            in_dim=join_dim,
            hidden=hidden_set,
            out_dim=hidden_set,
            num_layers=num_set_layers,
            dropout=dropout,
        )
        self.pred_mlp = SetEncoder(
            in_dim=pred_dim,
            hidden=hidden_set,
            out_dim=hidden_set,
            num_layers=num_set_layers,
            dropout=dropout,
        )

        final_in = 3 * hidden_set

        self.final_mlp = nn.Sequential(
            nn.Linear(final_in, hidden_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_out, hidden_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_out, hidden_out),
            nn.ReLU(),
            nn.Linear(hidden_out, 1),
        )

    def forward(self,
                tables_X, tables_m,
                joins_X, joins_m,
                preds_X, preds_m):

        T = self.table_mlp(tables_X, tables_m)
        J = self.join_mlp(joins_X, joins_m)
        P = self.pred_mlp(preds_X, preds_m)
        combined = torch.cat([T, J, P], dim=-1)
        out = self.final_mlp(combined).squeeze(-1)
        return out
