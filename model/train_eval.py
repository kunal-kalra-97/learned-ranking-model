# train_test_model.py
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from model.mscn_dataset import make_dataloaders
from model.mscn_model import MSCNModel


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        tables_X, tables_m, joins_X, joins_m, preds_X, preds_m, y = batch
        tables_X = tables_X.to(device)
        tables_m = tables_m.to(device)
        joins_X  = joins_X.to(device)
        joins_m  = joins_m.to(device)
        preds_X  = preds_X.to(device)
        preds_m  = preds_m.to(device)
        y        = y.to(device)

        optimizer.zero_grad()
        y_pred = model(tables_X, tables_m, joins_X, joins_m, preds_X, preds_m)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        tables_X, tables_m, joins_X, joins_m, preds_X, preds_m, y = batch
        tables_X = tables_X.to(device)
        tables_m = tables_m.to(device)
        joins_X  = joins_X.to(device)
        joins_m  = joins_m.to(device)
        preds_X  = preds_X.to(device)
        preds_m  = preds_m.to(device)
        y        = y.to(device)

        y_pred = model(tables_X, tables_m, joins_X, joins_m, preds_X, preds_m)
        loss = loss_fn(y_pred, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def train_model(batch_size=64, epochs=20, lr=1e-3, feature_dims: Tuple[int, int, int] = None):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    Ft, Fj, Fp = feature_dims
    train_loader, test_loader = make_dataloaders(
        batch_size = batch_size,
        feature_dims = feature_dims
    )

    model = MSCNModel(
        table_dim=Ft,
        join_dim=Fj,
        pred_dim=Fp,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


