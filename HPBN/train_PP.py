# train_hpbn_ppmi_3class_search_full_v2.py
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score
from itertools import product
import csv

from hbpn_model_PP import HPBN_ADNI

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
# Evaluation
# -----------------------------
def evaluate(model, loader, device):
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
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = 100.0 * accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        auc = 0.5
    f1 = f1_score(all_labels, all_preds, average="macro")

    cm = confusion_matrix(all_labels, all_preds)
    sens_list, spec_list = [], []
    num_classes = cm.shape[0]
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        sens_list.append(tp / (tp + fn + 1e-6))
        spec_list.append(tn / (tn + fp + 1e-6))
    sens = np.mean(sens_list)
    spec = np.mean(spec_list)

    avg_loss = total_loss / max(1, len(loader))
    return acc, auc, f1, sens, spec, avg_loss

# -----------------------------
# Single Run
# -----------------------------
def train_one_run(fcn_path, scn_path, label_path, device,
                  dim_proto_node, dim_proto_graph, dim_cls,
                  num_node_proto, num_graph_proto, num_class_proto):
    dataset = ADNIDataset(fcn_path, scn_path, label_path)
    N = len(dataset)
    indices = np.random.permutation(N)
    train_end = int(0.7 * N)
    val_end = int(0.85 * N)
    train_set = torch.utils.data.Subset(dataset, indices[:train_end])
    val_set = torch.utils.data.Subset(dataset, indices[train_end:val_end])
    test_set = torch.utils.data.Subset(dataset, indices[val_end:])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    model = HPBN_ADNI(
        node_in_dim=90,
        dim_proto_node=dim_proto_node,
        dim_proto_graph=dim_proto_graph,
        dim_cls=dim_cls,
        num_node_proto=num_node_proto,
        num_graph_proto=num_graph_proto,
        num_class_proto=num_class_proto,
        num_classes=3
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    max_epochs, patience = 50, 7
    best_val_loss, counter, best_state_dict = float("inf"), 0, None

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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    acc, auc, f1, sens, spec, _ = evaluate(model, test_loader, device)
    return acc, auc, f1, sens, spec

# -----------------------------
# Full Parameter Search with 5 runs
# -----------------------------
def full_param_search():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fcn_path = "./PPMI/final_matrix_fcn_PPMI.mat"
    scn_path = "./PPMI/final_matrix_scn_PPMI.mat"
    label_path = "./PPMI/final_matrix_labels_PPMI.mat"

    # 搜索空间
    node_dims = [32, 48, 64, 90]
    graph_dims = [32, 48, 64, 90]
    cls_dims = [16, 32, 48, 64]
    num_node_protos = [32, 64, 128]
    num_graph_protos = [16, 32, 64]
    num_class_protos = [8, 16, 32]

    results = []

    for n_dim, g_dim, c_dim, n_proto, g_proto, c_proto in product(
            node_dims, graph_dims, cls_dims, num_node_protos, num_graph_protos, num_class_protos):
        print(f"\n--- Testing n_dim={n_dim}, g_dim={g_dim}, c_dim={c_dim}, "
              f"n_proto={n_proto}, g_proto={g_proto}, c_proto={c_proto} ---")
        
        acc_list, auc_list, f1_list, sens_list, spec_list = [], [], [], [], []

        for run in range(5):  # 每组参数重复5次
            acc, auc, f1, sens, spec = train_one_run(
                fcn_path, scn_path, label_path, device,
                dim_proto_node=n_dim,
                dim_proto_graph=g_dim,
                dim_cls=c_dim,
                num_node_proto=n_proto,
                num_graph_proto=g_proto,
                num_class_proto=c_proto
            )
            acc_list.append(acc)
            auc_list.append(auc)
            f1_list.append(f1)
            sens_list.append(sens)
            spec_list.append(spec)

        results.append({
            "dim_proto_node": n_dim,
            "dim_proto_graph": g_dim,
            "dim_cls": c_dim,
            "num_node_proto": n_proto,
            "num_graph_proto": g_proto,
            "num_class_proto": c_proto,
            "Acc_mean": np.mean(acc_list),
            "Acc_std": np.std(acc_list),
            "AUC_mean": np.mean(auc_list),
            "AUC_std": np.std(auc_list),
            "F1_mean": np.mean(f1_list),
            "F1_std": np.std(f1_list),
            "Sens_mean": np.mean(sens_list),
            "Sens_std": np.std(sens_list),
            "Spec_mean": np.mean(spec_list),
            "Spec_std": np.std(spec_list)
        })

        print(f"Mean Acc: {np.mean(acc_list):.2f}%, Std: {np.std(acc_list):.2f}")

    # 输出 CSV
    keys = results[0].keys()
    with open("hpbn_param_search_results_5runs.csv", "w", newline="") as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    print("\nParameter search finished. Results saved to hpbn_param_search_results_5runs.csv")

if __name__ == "__main__":
    full_param_search()
