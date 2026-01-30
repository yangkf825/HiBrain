# hpbn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLayer(nn.Module):
    """
    通用 prototype 匹配层：
    - prototypes: [K, d]
    - 输入 x:
        [B, N, d] 或 [B, d]
    - 输出:
        matched: [B, N, d] 或 [B, d]
        scores:  [B, N, K] 或 [B, K]
    """
    def __init__(self, num_prototypes, dim):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.dim = dim
        # 原型初始化为较小范围
        self.prototypes = nn.Parameter(
            torch.empty(num_prototypes, dim).uniform_(-0.1, 0.1)
        )

    def forward(self, x, hard=True):
        """
        x: [B, N, d] or [B, d]
        hard=True  : 每个样本/节点选择 1 个最相似原型 (argmax)
        hard=False : 使用 softmax 权重，对全部原型加权求和 (可视作匹配到多个原型)
        """
        squeeze_back = False
        if x.dim() == 2:  # [B, d] → [B, 1, d]
            x = x.unsqueeze(1)
            squeeze_back = True

        B, N, d = x.shape
        # [B, N, 1, d] - [1, 1, K, d] → [B, N, K, d]
        diff = x.unsqueeze(2) - self.prototypes.view(1, 1, self.num_prototypes, d)
        dist_sq = (diff ** 2).sum(-1)   # [B, N, K]
        scores = -dist_sq               # 负欧氏距离，越大越相似

        if hard:
            # 只取一个最相似的
            idx = scores.argmax(-1)          # [B, N]
            matched = self.prototypes[idx]   # [B, N, d]
        else:
            # soft 匹配：用 softmax 权重对所有原型加权求和
            w = F.softmax(scores, dim=-1)    # [B, N, K]
            matched = torch.einsum("bnk,kd->bnd", w, self.prototypes)  # [B, N, d]

        if squeeze_back:
            matched = matched.squeeze(1)  # [B, d]
            scores = scores.squeeze(1)    # [B, K]

        return matched, scores


class HPBN_ADNI(nn.Module):
    """
    HPBN 三层原型结构，适配 ADNI 多模态脑图：

    输入:
        fcn: [B, 90, 90]   （fMRI / FCN）
        scn: [B, 90, 90]   （DTI / SCN）
        —— 每行看作一个 ROI 节点的 90 维特征，不在模型外拼接成 180×90。

    Level-1：Node-level prototypes
        - 对 fcn 的 90 个节点分别与 node-level prototypes 匹配
        - 对 scn 的 90 个节点分别与 node-level prototypes 匹配
        - 得到 N_f(g), N_s(g) 两组节点级原型集合
        - 再对每组做 pooling → E_G_f, E_G_s 作为图级向量（Level-2 输入）

    Level-2：Graph-level prototypes
        - E_G_f, E_G_s 分别与 graph-level prototypes 匹配 → G_f, G_s
        - 对 G_f, G_s 经过 MLP → E_C_f, E_C_s 作为 Level-3 输入

    Level-3：Class-level prototypes（数量可大于类别数）
        - E_C_f, E_C_s 与 class-level prototypes 做 soft 匹配（hard=False）
          → C_f, C_s 是由多个 class 原型加权得到的向量

    Final Fusion：
        - N_proto:  flatten(N_f, N_s) （两脑图所有节点匹配到的原型）
        - G_proto:  G_f, G_s
        - C_proto:  C_f, C_s
        - R(g) = [N_proto ∥ G_proto ∥ C_proto]
        - logits = classifier(R(g))
    """
    def __init__(
        self,
        node_in_dim=90,         # 每个节点特征维度（90×90 的一行）
        dim_proto_node=90,      # 节点级原型维度，默认与节点维度相同 → 直接匹配 (满足你第2点)
        dim_proto_graph=90,     # 图级原型维度（这里设成同一维度方便）
        dim_cls=64,             # 类别原型空间维度
        num_node_proto=128,      # Level-1 节点原型数
        num_graph_proto=32,     # Level-2 图原型数
        num_class_proto=16,      # Level-3 类原型数（可以 > 类别数）
        num_classes=2           # 最终分类类别数
    ):
        super().__init__()

        self.node_in_dim = node_in_dim
        self.dim_proto_node = dim_proto_node
        self.dim_proto_graph = dim_proto_graph
        self.dim_cls = dim_cls
        self.num_classes = num_classes

        # ---------- Level-1: Node-level prototypes ----------
        # 直接用节点原始向量（90维）与节点原型（90维）做匹配（对应你第2点）
        self.node_proto_layer = PrototypeLayer(num_node_proto, dim_proto_node)

        # ---------- Level-2: Graph-level prototypes ----------
        # 从节点级原型池化得到 E_G_f, E_G_s，维度为 dim_proto_node
        # 这里设 graph-level proto 维度与 node-level 相同，便于匹配
        self.graph_proto_layer = PrototypeLayer(num_graph_proto, dim_proto_graph)

        # E_C_f / E_C_s = MLP(G_f / G_s) → 作为第三层输入（对应你第4点）
        self.graph_mlp = nn.Sequential(
            nn.Linear(dim_proto_graph, dim_proto_graph),
            nn.ReLU(inplace=True),
            nn.Linear(dim_proto_graph, dim_cls)
        )

        # ---------- Level-3: Class-level prototypes ----------
        # class-level 原型数量可以大于类别数（对应你第5点）
        self.class_proto_layer = PrototypeLayer(num_class_proto, dim_cls)

        # ---------- Final classifier ----------
        # N_proto: 两个模态的所有节点级原型匹配结果
        #   - fcn: [B, 90, dim_proto_node]
        #   - scn: [B, 90, dim_proto_node]
        #   → flatten 后: 2 * 90 * dim_proto_node
        n_proto_dim = 2 * 90 * dim_proto_node

        # G_proto: G_f, G_s 各 dim_proto_graph
        g_proto_dim = 2 * dim_proto_graph

        # C_proto: C_f, C_s 各 dim_cls
        c_proto_dim = 2 * dim_cls

        final_dim = n_proto_dim + g_proto_dim + c_proto_dim

        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, fcn, scn):
        """
        fcn: [B, 90, 90]  fMRI 图
        scn: [B, 90, 90]  DTI  图
        """
        B, N_f, D_f = fcn.shape
        B2, N_s, D_s = scn.shape
        assert B == B2, "两个模态 batch 大小必须一致"
        assert N_f == 90 and N_s == 90, f"期望每个模态 90 个节点, got {N_f}, {N_s}"
        assert D_f == self.node_in_dim and D_s == self.node_in_dim, \
            f"节点特征维度应为 {self.node_in_dim}, got {D_f}, {D_s}"

        # ========== Level-1：节点级原型匹配 ==========
        # 直接用行向量做节点特征，不做 GNN、不做多模态融合
        # fcn_nodes: [B, 90, node_in_dim]
        # scn_nodes: [B, 90, node_in_dim]
        fcn_nodes = fcn
        scn_nodes = scn

        # 分别与 node-level prototypes 匹配
        # node_matched_f: [B, 90, dim_proto_node]
        # node_scores_f:  [B, 90, num_node_proto]
        node_matched_f, node_scores_f = self.node_proto_layer(fcn_nodes, hard=False)
        node_matched_s, node_scores_s = self.node_proto_layer(scn_nodes, hard=False)

        # N_proto：三层融合时要用的是“匹配到的原型”，不是输入向量
        # 这里我们把两组节点原型 flatten，直接送入 classifier（对应你第6点）
        N_proto_f_flat = node_matched_f.reshape(B, -1)  # [B, 90*dim_proto_node]
        N_proto_s_flat = node_matched_s.reshape(B, -1)  # [B, 90*dim_proto_node]
        N_proto = torch.cat([N_proto_f_flat, N_proto_s_flat], dim=-1)  # [B, 2*90*dim_proto_node]

        # 为了 Level-2，我们对每个模态做 pooling 得到图级向量（仍然是由原型平均而来）
        E_G_f = node_matched_f.mean(dim=1)  # [B, dim_proto_node]
        E_G_s = node_matched_s.mean(dim=1)  # [B, dim_proto_node]

        # ========== Level-2：图级原型匹配 ==========
        # 分别与 graph-level prototypes 匹配
        G_proto_f, graph_scores_f = self.graph_proto_layer(E_G_f, hard=False)  # [B, dim_proto_graph]
        G_proto_s, graph_scores_s = self.graph_proto_layer(E_G_s, hard=False)  # [B, dim_proto_graph]

        # 通过 MLP 得到 Level-3 输入 E_C_f, E_C_s
        E_C_f = self.graph_mlp(G_proto_f)   # [B, dim_cls]
        E_C_s = self.graph_mlp(G_proto_s)   # [B, dim_cls]

        # ========== Level-3：类别级原型匹配（允许 > 类别数 & 多原型匹配） ==========
        # 这里用 soft 匹配 (hard=False)，意味着一个样本可以“匹配到多个 class 原型”的加权组合
        C_proto_f, class_scores_f = self.class_proto_layer(E_C_f, hard=False)  # [B, dim_cls]
        C_proto_s, class_scores_s = self.class_proto_layer(E_C_s, hard=False)  # [B, dim_cls]

        # ========== Final Fusion：三层原型 N_proto, G_proto, C_proto（对应你第6点） ==========
        G_proto = torch.cat([G_proto_f, G_proto_s], dim=-1)  # [B, 2*dim_proto_graph]
        C_proto = torch.cat([C_proto_f, C_proto_s], dim=-1)  # [B, 2*dim_cls]

        R = torch.cat([N_proto, G_proto, C_proto], dim=-1)   # [B, final_dim]
        logits = self.classifier(R)                          # [B, num_classes]

        aux = {
            "node_matched_f": node_matched_f,
            "node_matched_s": node_matched_s,
            "node_scores_f": node_scores_f,
            "node_scores_s": node_scores_s,
            "E_G_f": E_G_f,
            "E_G_s": E_G_s,
            "G_proto_f": G_proto_f,
            "G_proto_s": G_proto_s,
            "graph_scores_f": graph_scores_f,
            "graph_scores_s": graph_scores_s,
            "E_C_f": E_C_f,
            "E_C_s": E_C_s,
            "C_proto_f": C_proto_f,
            "C_proto_s": C_proto_s,
            "class_scores_f": class_scores_f,
            "class_scores_s": class_scores_s,
            "N_proto": N_proto,
            "G_proto": G_proto,
            "C_proto": C_proto,
            "R": R,
        }

        return logits, aux
