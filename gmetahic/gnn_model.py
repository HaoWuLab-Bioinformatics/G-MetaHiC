import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class StandardGraphConv(nn.Module):


    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels


        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        std = math.sqrt(6.0 / (self.in_channels + self.out_channels))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj_matrix):

        # 先添加自连接，再做对称归一化 \hat{A} = A + I, \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}
        adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0), device=adj_matrix.device)
        # 计算度并避免除零
        degree = adj_matrix.sum(dim=1)
        degree = torch.clamp_min(degree, 1e-12)
        degree_inv_sqrt = degree.pow(-0.5)

        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        norm_adj = torch.mm(torch.mm(D_inv_sqrt, adj_matrix), D_inv_sqrt)

        # GCN计算: A_norm * X * W
        support = torch.mm(x, self.weight)
        output = torch.mm(norm_adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output




# ===== RoPE Multi-Head Attention  =====
class RoPEMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True, rope_base=10000.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.rope_base = rope_base

        # 显式的 Q/K/V 投影与输出投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    @staticmethod
    def _build_rope_cache(seq_len, dim, base, device, dtype):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("s,d->sd", t, inv_freq)  # [S, dim/2]
        cos = torch.cos(freqs).to(dtype=dtype)[None, None, :, :]  # [1,1,S,dim/2]
        sin = torch.sin(freqs).to(dtype=dtype)[None, None, :, :]
        return cos, sin

    @staticmethod
    def _apply_rope(x, cos, sin):
        # x: [B,H,S,D]; cos/sin: [1,1,S,D/2]
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_odd  * cos + x_even * sin
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd
        return x_rot

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):

        if not self.batch_first:

            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        B, S, E = query.shape
        H = self.num_heads
        D = self.head_dim


        q = self.q_proj(query).view(B, S, H, D).transpose(1, 2)  # [B,H,S,D]
        k = self.k_proj(key).view(B, S, H, D).transpose(1, 2)    # [B,H,S,D]
        v = self.v_proj(value).view(B, S, H, D).transpose(1, 2)  # [B,H,S,D]


        cos, sin = self._build_rope_cache(S, D, self.rope_base, q.device, q.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # 注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # [B,H,S,S]


        if key_padding_mask is not None:

            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.bool()
            mask = key_padding_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))


        if attn_mask is not None:
            if attn_mask.dim() == 2:

                attn_scores = attn_scores + attn_mask[None, None, :, :]
            elif attn_mask.dim() == 3:

                attn_scores = attn_scores + attn_mask[:, None, :, :]
            else:
                raise ValueError("the  shape of attn_mask should be [S,S] or [B,S,S]")

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, E)
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)


        return attn_output, (attn_weights if need_weights else None)








class EfficientGraphStructureLearner(nn.Module):


    def __init__(self, input_dim, hidden_dim=16, k_neighbors=6):  # 进一步减少参数
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors

        self.feature_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):

        num_nodes = x.size(0)


        h = self.feature_proj(x)
        h = F.normalize(h, p=2, dim=1)


        similarity_matrix = torch.mm(h, h.t())


        adj_matrix = torch.zeros_like(similarity_matrix)
        k = min(self.k_neighbors, num_nodes - 1)


        _, top_k_indices = torch.topk(similarity_matrix, k + 1, dim=1)

        for i in range(num_nodes):
            neighbors = top_k_indices[i][1:]
            adj_matrix[i, neighbors] = similarity_matrix[i, neighbors]


        adj_matrix = (adj_matrix + adj_matrix.t()) / 2


        row_sums = adj_matrix.sum(dim=1, keepdim=True)
        adj_matrix = adj_matrix / (row_sums + 1e-8)

        return adj_matrix




class LightweightGraphEncoder(nn.Module):

    def __init__(self, input_dim=20, hidden_dim=32, output_dim=128,
                 num_layers=3, dropout=0.1):
        super().__init__()

        self.graph_learner = EfficientGraphStructureLearner(input_dim, hidden_dim=16)

        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.gcn_layers.append(StandardGraphConv(input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 2):
            self.gcn_layers.append(StandardGraphConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.gcn_layers.append(StandardGraphConv(hidden_dim, output_dim))
        self.norms.append(nn.LayerNorm(output_dim))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.res_input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.res_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.res_output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):

        # 学习图结构
        adj_matrix = self.graph_learner(x)

        h = x
        for i, (gcn, norm) in enumerate(zip(self.gcn_layers, self.norms)):
            h_prev = h

            h_new = gcn(h_prev, adj_matrix)
            h_new = norm(h_new)
            h_new = self.activation(h_new)

            if i == 0:
                skip = self.res_input(h_prev)
            elif i == len(self.gcn_layers) - 1:
                skip = self.res_output(h_prev)
            else:
                skip = self.res_hidden(h_prev)

            h = h_new + skip

            if i < len(self.gcn_layers) - 1:
                h = self.dropout(h)
        return h



class MemoryEfficientGraphBranchCov(nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.use_gcn = True
        self.sample_size = 256

        self.graph_encoder = LightweightGraphEncoder(
            input_dim=20,
            hidden_dim=32,
            output_dim=128,
            num_layers=3,
            dropout=dropout
        )

        self.seq_adapter = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.seq_adapter_res = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.channel_reduce = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.channel_skip = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=1)
        )

        self.head = nn.Sequential(


            nn.Linear(32 * self.sample_size, 512)
        )

    def forward(self, x, x_pb):

        B, C, L = x.shape
        batch_features = []
        for b in range(B):

            metacell_features = x[b].t()

            if L > self.sample_size:
                idx = torch.linspace(0, L - 1, steps=self.sample_size, device=x.device).round().long()
                sampled = metacell_features[idx]  # (S, 20)
            else:
                pad_len = self.sample_size - L
                if pad_len > 0:
                    pad = torch.zeros(pad_len, metacell_features.size(1), device=x.device, dtype=x.dtype)
                    sampled = torch.cat([metacell_features, pad], dim=0)  # (S, 20)
                else:
                    sampled = metacell_features  # (S, 20)

            if self.use_gcn:
                node_feats = self.graph_encoder(sampled)  # (S, 128)
            else:

                if sampled.size(1) >= 128:
                    node_feats = sampled[:, :128]
                else:
                    node_feats = F.pad(sampled, (0, 128 - sampled.size(1)))



            adapted_base = self.seq_adapter(node_feats)
            adapted_skip = self.seq_adapter_res(node_feats)
            adapted = adapted_base + adapted_skip
            batch_features.append(adapted)

        seq = torch.stack(batch_features, dim=0)
        seq = seq.transpose(1, 2)

        seq_red_base = self.channel_reduce(seq)
        seq_red_skip = self.channel_skip(seq)
        seq_reduced = seq_red_base + seq_red_skip
        out = self.head(seq_reduced.reshape(B, -1))
        return out




class MemoryEfficientGraphBranchPBulk(nn.Module):




    def __init__(self, dropout=0.1, in_channels=2, d_model=16, target_len=512):
        super().__init__()
        self.d_model = d_model
        self.target_len = target_len



        self.use_transformer = True

        self.pbulk_1d = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=11, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=7, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 32, kernel_size=5, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # 保持长度的一组卷积（含膨胀）
            nn.Conv1d(32, 32, kernel_size=5, stride=1, dilation=1,  padding="same"),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, dilation=1,  padding="same"),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, dilation=2,  padding="same"),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, dilation=3,  padding="same"),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, dilation=5,  padding="same"),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, dilation=5,  padding="same"),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, dilation=7,  padding="same"),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, dilation=11, padding="same"),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, dilation=11, padding="same"),
            nn.BatchNorm1d(32), nn.ReLU(),

            nn.MaxPool1d(kernel_size=5),

            nn.Conv1d(32, 16, kernel_size=3, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),

            nn.Conv1d(16, 16, kernel_size=3, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, dilation=1, padding="same"),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Dropout(dropout),
        )


        self.to_512 = nn.AdaptiveAvgPool1d(target_len)


        self.pos_encoding = PositionalEncoding(d_model, max_len=target_len)
        self.transformer_blocks = nn.ModuleList([
            LightTransformerBlock(d_model, nhead=4, dropout=dropout)
            for _ in range(3)
        ])


        self.head = nn.Sequential(
            nn.Linear(16* 512 , 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048,1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,512)
        )

    def forward(self, x2):

        x = self.pbulk_1d(x2)
        x = self.to_512(x)


        seq = x.transpose(1, 2)

        if self.use_transformer:
            seq = self.pos_encoding(seq)
            for block in self.transformer_blocks:
                seq = block(seq)
        else:
            pass


        feat = seq.transpose(1, 2)
        feat = torch.flatten(feat,1)
        out = self.head(feat)
        return out


class LightTransformerBlock(nn.Module):


    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()

        self.self_attn = RoPEMultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1024):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):

        return x


#### 双向交叉注意力
class MemoryEfficientCrossAttentionFusion(nn.Module):

    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()

        self.cross12 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross21 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)


        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )


        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feat1, feat2):


        f1 = feat1.unsqueeze(1)
        f2 = feat2.unsqueeze(1)


        attn12, _ = self.cross12(f1, f2, f2)
        attn21, _ = self.cross21(f2, f1, f1)


        attn12 = attn12.squeeze(1)
        attn21 = attn21.squeeze(1)


        gate_logits = self.gate(torch.cat([feat1, feat2], dim=1))
        alpha = torch.softmax(gate_logits, dim=1)
        w12 = alpha[:, :1]
        w21 = alpha[:, 1:2]
        attn_bi = w12 * attn12 + w21 * attn21


        fused_ctx = self.ffn(torch.cat([feat1, feat2], dim=1))
        out = self.norm(attn_bi + fused_ctx)
        return out


class MemoryEfficientGraphTrunk(nn.Module):


    def __init__(self, branch_pbulk, branch_cov, dropout=0.1):
        super().__init__()

        self.branch_pbulk = branch_pbulk
        self.branch_cov = branch_cov

        self.use_cross_attn = True

        self.simple_fuse = nn.Sequential(
            nn.Linear(2 * 512 , 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,200)
        )



        self.cross_fusion = MemoryEfficientCrossAttentionFusion(512, nhead=4, dropout=dropout)


        self.output_net = nn.Sequential(
            nn.Linear(512 * 3, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 200)
        )

    def forward(self, x, x2):

        cov_features = self.branch_cov(x, x2)  # (B, 512)
        # with torch.no_grad():
        #     cov_features = self.branch_cov(x, x2)
        pbulk_features = self.branch_pbulk(x2)  # (B, 512)

        if self.use_cross_attn:

            cross_fused = self.cross_fusion(pbulk_features, cov_features)

            final_features = torch.cat([pbulk_features, cov_features, cross_fused], dim=1)

            output = self.output_net(final_features)
        else:
            fused_branch = torch.cat([pbulk_features,cov_features],dim=1)
            output = self.simple_fuse(fused_branch)

        return output


def create_gmetahic(dropout=0.1):

    branch_pbulk = MemoryEfficientGraphBranchPBulk(dropout=dropout)
    branch_cov = MemoryEfficientGraphBranchCov(dropout=dropout)
    model = MemoryEfficientGraphTrunk(branch_pbulk, branch_cov, dropout=dropout)

    return model


def initialize_model_weights(model):

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.MultiheadAttention):
            nn.init.xavier_normal_(m.in_proj_weight)
            nn.init.xavier_normal_(m.out_proj.weight)
