import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight



def model_00(train_loader,
             val_loader,
             input_size,
             hidden_size=64,
             lr=0.001,
             epochs=10,
             device=None):
    """
    Simple 2‑layer MLP for 9‑class classification.
    Returns
    -------
    model   : trained torch.nn.Module
    history : dict with keys
              'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim

    # -------- device ---------
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------- model ---------
    class BasicMLP(nn.Module):
        def __init__(self, inp, hid):
            super().__init__()
            self.fc1 = nn.Linear(inp, hid)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hid, 9)          # 9 classes

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)

    model = BasicMLP(input_size, hidden_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------- history trackers ---------
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # -------- training loop ------------
    for epoch in range(1, epochs + 1):
        # --- TRAIN ---
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out   = model(x)
            loss  = criterion(out, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total   += x.size(0)

        train_loss = loss_sum / total
        train_acc  = 100 * correct / total
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)

        # --- VALIDATION ---
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out  = model(x)
                loss_sum += criterion(out, y).item() * x.size(0)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total   += x.size(0)

        val_loss = loss_sum / total
        val_acc  = 100 * correct / total
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)

        print(f"[Epoch {epoch:02d}/{epochs}] "
              f"Train: loss {train_loss:.4f} | acc {train_acc:.2f}%   "
              f"Val: loss {val_loss:.4f} | acc {val_acc:.2f}%")

    return model, hist


################################################################################################################
def model_01(train_loader, val_loader, input_size,
             hidden_size=128, lr=1e-3, epochs=30,
             num_classes=9, seed=42, device=None):
    """
    Deeper MLP with class‑weighted CE
    Returns: model, history  (history has loss and accuracy arrays)
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- model ----------
    class BasicMLP(nn.Module):
        def __init__(self, inp, hid, ncls):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(inp, hid),
                nn.BatchNorm1d(hid),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hid, hid // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hid // 2, ncls)
            )
        def forward(self, x): return self.net(x)

    model = BasicMLP(input_size, hidden_size, num_classes).to(device)

    # ---------- class weights ----------
    from sklearn.utils.class_weight import compute_class_weight
    all_labels = torch.cat([b[1] for b in train_loader]).cpu().numpy()
    w = compute_class_weight('balanced', classes=np.arange(num_classes), y=all_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---------- history trackers ----------
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs+1):
        # ======== TRAIN ========
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()

            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = loss_sum / total
        train_acc  = 100 * correct / total
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)

        # ======== VAL ========
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss_sum += criterion(out, y).item() * x.size(0)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)

        val_loss = loss_sum / total
        val_acc  = 100 * correct / total
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)

        print(f"[Epoch {epoch:02d}/{epochs}] "
              f"Train: loss {train_loss:.4f} | acc {train_acc:.2f}%   "
              f"Val: loss {val_loss:.4f} | acc {val_acc:.2f}%")

    return model, hist


