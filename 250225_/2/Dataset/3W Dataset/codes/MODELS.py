import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def model_00(train_loader, val_loader, input_size, hidden_size=64, lr=0.001, epochs=10):
    class BasicMLP(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(BasicMLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_size, 9)  # multiclass classification (2 classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # Initialize model, loss, optimizer
    model = BasicMLP(input_size=input_size, hidden_size=hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {running_loss:.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%\n")

    return model

###########################################################################################################################################################################
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

def model_01(train_loader,
            val_loader,
            input_size,
            hidden_size=128,
            lr=1e-3,
            epochs=30,
            num_classes=9,
            seed=42,
            device=None):
    """
    Multiclass MLP with class‑balanced CrossEntropyLoss.
    """

    # ---------------------- reproducibility ----------------------
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------------------- device -------------------------------
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------------------- model -------------------------------
    class BasicMLP(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size // 2, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    model = BasicMLP(input_size, hidden_size, num_classes).to(device)

    # ---------------------- class weights -----------------------
    # gather labels from train_loader once
    all_train_labels = torch.cat([batch[1] for batch in train_loader]).cpu().numpy()
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=all_train_labels
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---------------------- training loop -----------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_total   += labels.size(0)
            running_correct += (preds == labels).sum().item()

        train_loss = running_loss / running_total
        train_acc  = 100 * running_correct / running_total

        # ---------------- validation ----------------------------
        model.eval()
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss_sum += criterion(outputs, labels).item() * inputs.size(NN0)
                _, preds = torch.max(outputs, 1)
                val_total   += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss = val_loss_sum / val_total
        val_acc  = 100 * val_correct / val_total

        print(f"[Epoch {epoch:02d}/{epochs}] "
              f"Train: loss {train_loss:.4f} | acc {train_acc:.2f}%   "
              f"Val: loss {val_loss:.4f} | acc {val_acc:.2f}%")

    return model
