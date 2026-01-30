# hbpn_model_PP.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeLayer(nn.Module):
    """
    通用 prototype 匹配层：
    - prototypes: [K, d]
    - 输入 x: [B, N, d] 或 [B, d]
    - 输出:
        matched: [B, N, d] 或 [B, d]
        scores:  [B, N, K] 或 [B, K]
    """
    def __init__(self, num_prototypes, dim):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.dim = dim
        self.prototypes = nn.Parameter(
            torch.empty(num_prototypes, dim).uniform_(-0.1, 0.1)
        )

    def forward(self, x, hard=True):
        squeeze_back = False
        if x.dim() == 2:  # [B, d] → [B, 1, d]
            x = x.unsqueeze(1)
            squeeze_back = True

        B, N, d = x.shape
        diff = x.unsqueeze(2) - self.prototypes.view(1, 1, self.num_prototypes, d)
        dist_sq = (diff ** 2).sum(-1)  # [B, N, K]
        scores = -dist_sq

        if hard:
            idx = scores.argmax(-1)          # [B, N]
            matched = self.prototypes[idx]   # [B, N, d]
        else:
            w = F.softmax(scores, dim=-1)    # [B, N, K]
            matched = torch.einsum("bnk,kd->bnd", w, self.prototypes)  # [B, N, d]

        if squeeze_back:
            matched = matched.squeeze(1)
            scores = scores.squeeze(1)

        return matched, scores


class HPBN_ADNI(nn.Module):
    """
    HPBN 三层原型结构，带显式 embedding
    """
    def __init__(
        self,
        node_in_dim=90,        # 原始节点特征维度
        dim_proto_node=48,     # 节点 embedding / 原型维度
        dim_proto_graph=32,    # 图级 embedding / 原型维度
        dim_cls=16,            # 类别 embedding / 原型维度
        num_node_proto=64,
        num_graph_proto=32,
        num_class_proto=16,
        num_classes=3
    ):
        super().__init__()

        self.node_in_dim = node_in_dim
        self.dim_proto_node = dim_proto_node
        self.dim_proto_graph = dim_proto_graph
        self.dim_cls = dim_cls
        self.num_classes = num_classes

        # ---------- Node-level embedding & prototypes ----------
        self.node_embedding = nn.Linear(node_in_dim, dim_proto_node)
        self.node_proto_layer = PrototypeLayer(num_node_proto, dim_proto_node)

        # ---------- Graph-level embedding & prototypes ----------
        self.graph_embedding = nn.Linear(dim_proto_node, dim_proto_graph)
        self.graph_proto_layer = PrototypeLayer(num_graph_proto, dim_proto_graph)
        self.graph_mlp = nn.Sequential(
            nn.Linear(dim_proto_graph, dim_proto_graph),
            nn.ReLU(inplace=True),
            nn.Linear(dim_proto_graph, dim_cls)
        )

        # ---------- Class-level prototypes ----------
        self.class_proto_layer = PrototypeLayer(num_class_proto, dim_cls)

        # ---------- Final classifier ----------
        n_proto_dim = 2 * 90 * dim_proto_node
        g_proto_dim = 2 * dim_proto_graph
        c_proto_dim = 2 * dim_cls
        final_dim = n_proto_dim + g_proto_dim + c_proto_dim

        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, fcn, scn):
        B, N_f, D_f = fcn.shape
        B2, N_s, D_s = scn.shape
        assert B == B2
        assert N_f == 90 and N_s == 90
        assert D_f == self.node_in_dim and D_s == self.node_in_dim

        # ---------- Node-level embedding ----------
        fcn_nodes = self.node_embedding(fcn)  # [B, 90, dim_proto_node]
        scn_nodes = self.node_embedding(scn)

        # Node-level prototype matching
        node_matched_f, node_scores_f = self.node_proto_layer(fcn_nodes, hard=False)
        node_matched_s, node_scores_s = self.node_proto_layer(scn_nodes, hard=False)

        # Flatten node-level protos
        N_proto = torch.cat([
            node_matched_f.reshape(B, -1),
            node_matched_s.reshape(B, -1)
        ], dim=-1)

        # Graph-level embedding
        E_G_f = self.graph_embedding(node_matched_f.mean(dim=1))  # [B, dim_proto_graph]
        E_G_s = self.graph_embedding(node_matched_s.mean(dim=1))

        # Graph-level prototype matching
        G_proto_f, _ = self.graph_proto_layer(E_G_f, hard=False)
        G_proto_s, _ = self.graph_proto_layer(E_G_s, hard=False)

        # MLP → class-level embedding
        E_C_f = self.graph_mlp(G_proto_f)
        E_C_s = self.graph_mlp(G_proto_s)

        # Class-level prototype matching
        C_proto_f, _ = self.class_proto_layer(E_C_f, hard=False)
        C_proto_s, _ = self.class_proto_layer(E_C_s, hard=False)

        # ---------- Final fusion ----------
        G_proto = torch.cat([G_proto_f, G_proto_s], dim=-1)
        C_proto = torch.cat([C_proto_f, C_proto_s], dim=-1)
        R = torch.cat([N_proto, G_proto, C_proto], dim=-1)

        logits = self.classifier(R)

        aux = {
            "node_matched_f": node_matched_f,
            "node_matched_s": node_matched_s,
            "E_G_f": E_G_f,
            "E_G_s": E_G_s,
            "G_proto_f": G_proto_f,
            "G_proto_s": G_proto_s,
            "E_C_f": E_C_f,
            "E_C_s": E_C_s,
            "C_proto_f": C_proto_f,
            "C_proto_s": C_proto_s,
            "R": R
        }

        return logits, aux
