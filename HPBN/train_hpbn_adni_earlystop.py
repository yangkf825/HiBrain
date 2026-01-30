# train_hpbn_adni_binary_threshold_earlystop.py
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from hpbn_model import HPBN_ADNI


# -----------------------------
# Dataset
# -----------------------------
class ADNIDataset(Dataset):
    def __init__(self, fcn_path, scn_path, label_path):
        fcn = sio.loadmat(fcn_path)
        scn = sio.loadmat(scn_path)
        label = sio.loadmat(label_path)

        self.fcn = np.array(fcn["fcn"], dtype=np.float32)      # [N, 90, 90]
        self.scn = np.array(scn["scn"], dtype=np.float32)      # [N, 90, 90]
        self.labels = np.array(label["label"], dtype=np.int64).squeeze()  # [N]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        f = torch.tensor(self.fcn[idx])
        s = torch.tensor(self.scn[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return f, s, y


# -----------------------------
# Threshold Search (Validation)
# -----------------------------
def search_best_threshold(labels, probs):
    best_th = 0.5
    best_score = -1.0

    for th in np.arange(0.1, 0.91, 0.01):
        preds = (probs >= th).astype(int)
        cm = confusion_matrix(labels, preds)

        if cm.shape != (2, 2):
            continue

        TN, FP, FN, TP = cm.ravel()
        sens = TP / (TP + FN + 1e-6)
        spec = TN / (TN + FP + 1e-6)

        balanced_acc = 0.5 * (sens + spec)

        if balanced_acc > best_score:
            best_score = balanced_acc
            best_th = th

    return best_th, best_score


# -----------------------------
# Evaluation (Argmax, for Val)
# -----------------------------
def evaluate(model, loader, device, return_probs=False):
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for fcn, scn, y in loader:
            fcn, scn, y = fcn.to(device), scn.to(device), y.to(device)

            logits, _ = model(fcn, scn)
            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = 100.0 * (all_preds == all_labels).mean()
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, average="binary")

    cm = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = cm.ravel()
    sens = TP / (TP + FN + 1e-6)
    spec = TN / (TN + FP + 1e-6)

    avg_loss = total_loss / max(1, len(loader))

    if return_probs:
        return acc, auc, f1, sens, spec, avg_loss, all_probs, all_labels
    else:
        return acc, auc, f1, sens, spec, avg_loss


# -----------------------------
# Evaluation (Fixed Threshold)
# -----------------------------
def evaluate_with_threshold(model, loader, device, threshold):
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for fcn, scn, y in loader:
            fcn, scn, y = fcn.to(device), scn.to(device), y.to(device)

            logits, _ = model(fcn, scn)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = 100.0 * (all_preds == all_labels).mean()
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, average="binary")

    cm = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = cm.ravel()
    sens = TP / (TP + FN + 1e-6)
    spec = TN / (TN + FP + 1e-6)

    return acc, auc, f1, sens, spec


# -----------------------------
# Single Run (with Early Stopping)
# -----------------------------
def train_one_run(fcn_path, scn_path, label_path, device):
    dataset = ADNIDataset(fcn_path, scn_path, label_path)
    N = len(dataset)
    indices = np.random.permutation(N)

    train_end = int(0.7 * N)
    val_end = int(0.85 * N)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    model = HPBN_ADNI(
        node_in_dim=90,
        dim_proto_node=90,
        dim_proto_graph=90,
        dim_cls=64,
        num_node_proto=64,
        num_graph_proto=32,
        num_class_proto=16,
        num_classes=2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # -------- Early Stopping Params --------
    max_epochs = 50
    patience = 3
    counter = 0
    best_val_loss = float("inf")
    best_state_dict = None

    # -------- Train --------
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for fcn, scn, y in train_loader:
            fcn, scn, y = fcn.to(device), scn.to(device), y.to(device)

            logits, _ = model(fcn, scn)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc, _, _, _, _, val_loss = evaluate(model, val_loader, device)

        # ---- Early Stopping Logic ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            counter = 0
        else:
            counter += 1

        print(f"Epoch [{epoch+1}/{max_epochs}] "
              f"Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # -------- Load Best Model --------
    model.load_state_dict(best_state_dict)

    # -------- Validation Threshold Search --------
    _, _, _, _, _, _, val_probs, val_labels = evaluate(
        model, val_loader, device, return_probs=True
    )

    best_th, best_bal_acc = search_best_threshold(val_labels, val_probs)
    print(f"Best Threshold (Val): {best_th:.2f}, "
          f"Balanced Acc: {best_bal_acc:.4f}")

    # -------- Test --------
    acc, auc, f1, sens, spec = evaluate_with_threshold(
        model, test_loader, device, best_th
    )

    return acc, auc, f1, sens, spec


# -----------------------------
# 5 Runs Experiment
# -----------------------------
def run_5_experiments():
    device = "cuda" if torch.cuda.is_available() else "cpu"

#    fcn_path = "./ADNI/final_matrix_fcn_NC-MCI.mat"
#    scn_path = "./ADNI/final_matrix_scn_NC-MCI.mat"
#    label_path = "./ADNI/final_matrix_labels_NC-MCI.mat"
#    fcn_path = "./ADNI/final_matrix_fcn_AD-MCI.mat"
#    scn_path = "./ADNI/final_matrix_scn_AD-MCI.mat"
#    label_path = "./ADNI/final_matrix_labels_AD-MCI.mat"
    fcn_path = "./ADNI/final_matrix_fcn_AD-NC.mat"
    scn_path = "./ADNI/final_matrix_scn_AD-NC.mat"
    label_path = "./ADNI/final_matrix_labels_AD-NC.mat"

    ACC, AUC, F1, SENS, SPEC = [], [], [], [], []

    for i in range(5):
        print(f"\n===== 实验 {i+1} / 5 =====")
        acc, auc, f1, sens, spec = train_one_run(
            fcn_path, scn_path, label_path, device
        )

        ACC.append(acc)
        AUC.append(auc)
        F1.append(f1)
        SENS.append(sens)
        SPEC.append(spec)

        print(f"[Run {i+1}] Test Acc: {acc:.2f}%")
        print(f"   AUC: {auc:.4f}, F1: {f1:.4f}")
        print(f"   Sensitivity: {sens:.4f}, Specificity: {spec:.4f}")

    print("\n===== 5 次实验结果统计 (Test) =====")
    print(f"Accuracy:    {np.mean(ACC):.4f} ± {np.std(ACC):.4f}")
    print(f"AUC:         {np.mean(AUC):.4f} ± {np.std(AUC):.4f}")
    print(f"F1:          {np.mean(F1):.4f} ± {np.std(F1):.4f}")
    print(f"Sensitivity: {np.mean(SENS):.4f} ± {np.std(SENS):.4f}")
    print(f"Specificity: {np.mean(SPEC):.4f} ± {np.std(SPEC):.4f}")


if __name__ == "__main__":
    run_5_experiments()
