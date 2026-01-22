import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
from layers import GraphConvolution
import math

class SimilarityFunctions(nn.Module):
    def __init__(self, sigma=1.0):
        """
        初始化不同的相似度类型
        :param similarity_type: 'cosine', 'gaussian' 或 'dot'
        :param sigma: 高斯相似度的参数 (只有高斯相似度需要)
        """
        super(SimilarityFunctions, self).__init__()
        self.sigma = sigma

    def forward(self, query, history):
        """
        根据选择的相似度类型计算相似度
        :param query: 当前查询向量
        :param history: 历史向量
        :return: 相似度值
        """
        return self.gaussian_similarity(query, history)

    def gaussian_similarity(self, query, history):
        """
        计算高斯相似度
        :param query: 当前查询向量
        :param history: 历史向量
        :return: 高斯相似度
        """
        dist = torch.norm(query - history, p=2, dim=-1)  # 欧氏距离
        return torch.exp(-dist ** 2 / (2 * self.sigma ** 2))

# 频率最小阈值是1
class FrequencyLayer(nn.Module):
    def __init__(self, d_model, low_freq_ratio=0.3, mode='full'):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

        # 可学习的频率阈值，帮助决定低频成分的截止频率。
        self.threshold_param = nn.Parameter(torch.tensor(low_freq_ratio))
        # 可学习的参数，初始化为随机值。它在频率处理过程中用来调整高频部分对最终输出的影响。
        self.beta = nn.Parameter(torch.randn(1, 1, 1))
        self.mode = mode

    def get_frequency_threshold(self, seq_len):
        """动态计算频率阈值"""
        # 计算动态频率阈值。根据输入的序列长度seq_len,计算出一个截止频率threshold;
        # 使用sigmoid函数对self.threshold_param进行变换，将其映射到0-1之间。
        # ？频率的对称性-没有搞明白-继续搞
        threshold = torch.sigmoid(self.threshold_param) * seq_len * 0.5
        return threshold.long().clamp(min=1, max=seq_len // 2)

    def forward(self, x, delta_t=None):
        """
        x: (B, L, D) 输入序列
        返回: 处理后的单个序列 (B, L, D)
        """
        B, L, D = x.shape  # 获取 batch_size, sequence_length 和 feature_dimension

        # 动态计算c值（可基于delta_t调整）
        # c表示低频和高频的分界线，也就是你希望低频成分和高频成分的"截止频率"，它决定了在哪个频率点上，信号从低频转为高频；
        if delta_t is not None and len(delta_t) > 1:
            # 如果提供了delta_t，计算delta_t的均值avg_interval，并基于它通过sigmoid函数来
            # 计算一个自适应的比例adaptive_ratio
            avg_interval = delta_t.mean()
            adaptive_ratio = torch.sigmoid(1.0 / (avg_interval + 1e-6)) * 0.5
            c = int(adaptive_ratio * L)
        else:
            c = self.get_frequency_threshold(L)

        # 确保c在合理范围内
        c = max(1, min(c, L - 1))

        # 转成 FrequencyLayer 期望的形状 (B, L, D) -> (B, 1, L, D)
        # 符合频率处理的要求
        input_tensor = x.unsqueeze(1)  # (B, 1, L, D)

        # FrequencyLayer 的核心处理逻辑
        # 进行快速傅里叶变换(x_fft通过torch.fft.rfft在序列长度维度dim=2上进行FFT，得到频域表示)
        # ?norm='ortho'是什么意思？
        x_fft = torch.fft.rfft(input_tensor, dim=2, norm='ortho')  # FFT in sequence length dimension
        low_pass = x_fft.clone() # x_fft的副本；
        # low_pass 是 x_fft 的副本，并将高于阈值 c 的频率成分置为 0，保留低频部分。
        # 从序列的频率成分中选择索引大于等于c的部分，保留了高于阈值c的频率成分。
        low_pass[:, :, c:, :] = 0  # 设定低频阈值
        # 将低频部分转换为时域;
        low_pass = torch.fft.irfft(low_pass, n=L, dim=2, norm='ortho')  # 进行逆FFT
        # 原始输入减去低频部分，得到高频成分;
        high_pass = input_tensor - low_pass
        # 最后，低频成分与高频成分结合，生成最终的输出;
        # === 根据 mode 决定用什么 ===
        if self.mode == 'full':
            # 原始做法：低频 + β² * 高频
            sequence_emb_fft = low_pass + (self.beta ** 2) * high_pass
        elif self.mode == 'low':
            # 仅低频
            sequence_emb_fft = low_pass
        elif self.mode == 'high':
            # 仅高频
            sequence_emb_fft = (self.beta ** 2) * high_pass
        elif self.mode == 'none':
            # 完全不用 FFT，直接返回原始输入
            sequence_emb_fft = input_tensor
        else:
            raise ValueError(f"Unknown FFT mode: {self.mode}")

        return sequence_emb_fft.squeeze(1)  # (B, L, D)

class TimeDecay(nn.Module):
    """
    时间衰减模块：
    - τ 表示衰减时间常数（单位：年）
    - Δt 自动转换成年，按相对当前时间计算
    - 根据统计结果初始化 τ≈0.5 年，范围 [0.1, 3.0]
    """
    def __init__(self, init_tau_years=0.5, time_scale=365.0, tau_min=0.1, tau_max=3.0):
        super().__init__()
        self.time_scale = time_scale # 时间缩放因子，将天转换为年
        self.tau_min = tau_min # τ的最小值（年）
        self.tau_max = tau_max # τ的最大值（年）

        # τ 参数化为 sigmoid 区间映射，保证可学习性和稳定性
        # 将初始τ值映射到[0,1]区间：(0.5-0.1)/(3.0-0.1) ≈ 0.1379
        init_ratio = (init_tau_years - tau_min) / (tau_max - tau_min)
        # 将数值裁剪到(0,1)区间避免边界问题
        init_ratio = np.clip(init_ratio, 1e-3, 1-1e-3)
        # 只是为了给可学习参数 tau_raw 设置一个合理的初始值，训练过程中这个值会被梯度更新改变，从而让模型学习到最佳的衰减时间常数。
        # torch.logit(torch.tensor(init_ratio, dtype=torch.float))的值域在（－∞，＋∞）
        # tau_raw 可以在整个实数范围自由学习,并在后续通过 sigmoid(tau_raw) 自动映射回 [0,1] 区间
        self.tau_raw = nn.Parameter(torch.logit(torch.tensor(init_ratio, dtype=torch.float)))

    def forward(self, delta_t):
        """
        delta_t: (batch, seq, 1)，单位 = 天（数值越大表示越新）
        返回衰减系数 (batch, seq, 1)
        """
        # τ（年）范围映射
        # 限定在最小值和最大值之间
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.tau_raw)

        # 反向处理：Δt越大越新 → 转换为“距当前”的相对时间差
        delta_t_max, _ = torch.max(delta_t, dim=1, keepdim=True)
        delta_t_rel = delta_t_max - delta_t

        # 将时间差从天转换为年
        delta_t_years = delta_t_rel / self.time_scale

        # 使用指数衰减公式
        # .clamp(min=1e-6): 确保衰减系数不为0，避免数值问题
        # 可学习的指数衰减常数tau
        # 自动时间标准化
        # 指数衰减
        # 通过clamp避免除零和数值下溢
        decay = torch.exp(-delta_t_years / tau).clamp(min=1e-6)
        return decay

class SimilarityDecay(nn.Module):
    def __init__(self, decay_rate=1.0):
        super(SimilarityDecay, self).__init__()
        self.decay_rate = decay_rate  # 衰减速率参数 λ

    def forward(self, similarity):
        """
        similarity: 相似度值，取值范围 [0, 1]
        返回一个衰减因子
        """
        # 归一化后的相似度，确保在 [0, 1] 范围
        similarity = torch.clamp(similarity, min=0.0, max=1.0)

        # 使用指数衰减公式
        decay = torch.exp(-self.decay_rate * (1 - similarity))  # 相似度越高衰减因子越小
        return decay

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class PatientEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device("cpu:0"), fft_mode='full'):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.fft_mode = fft_mode

        # === Embeddings ===
        self.emb_diag = nn.Embedding(vocab_size[0], emb_dim)
        self.emb_proc = nn.Embedding(vocab_size[1], emb_dim)
        self.emb_drug = nn.Embedding(vocab_size[2], emb_dim * 2)
        self.dropout = nn.Dropout(0.4)

        # === Transformer Encoder ===
        # HAHA
        # self.transformerlayer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(self.transformerlayer, num_layers=1)
        self.linear_layer = nn.Linear(emb_dim * 1, emb_dim * 2)

        self.treatment = nn.Linear(emb_dim * 3, emb_dim * 2)

        self.query = nn.Linear(emb_dim * 4, emb_dim)

        self.time_decay = TimeDecay(init_tau_years=2.0, time_scale=365.0).to(device)

        if fft_mode == 'none':
            self.decompose = None
        else:
            self.decompose = FrequencyLayer(d_model=emb_dim, low_freq_ratio=0.3, mode=fft_mode)

        # === Add Multihead Attention Layer ===
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim * 4, num_heads=4, batch_first=True)

    def forward(self, input):
        diag_seq, proc_seq, drug_seq, delta_t_seq = [], [], [], []

        def mean_emb(emb_layer, idxs):
            if len(idxs) == 0:
                # 没有药物时，给一个零向量
                return torch.zeros((1, 1, self.emb_dim * 2), device=self.device)
            x = self.dropout(emb_layer(torch.LongTensor(idxs).to(self.device)))
            return x.mean(dim=0, keepdim=True).unsqueeze(0)  # (1,1,dim)

        prev_drugs = []  # 累计之前所有药物
        # === 序列展开 ===
        for adm in input:
            diag_seq.append(mean_emb(self.emb_diag, adm[0]))
            proc_seq.append(mean_emb(self.emb_proc, adm[1]))
            # ✅ 统计截至 t-1 的药物
            drug_seq.append(mean_emb(self.emb_drug, prev_drugs))
            # 更新累计药物（加入当前次药物）
            prev_drugs.extend(adm[2])
            delta_t_seq.append(torch.tensor([adm[-1]], dtype=torch.float).to(self.device))

        diag_seq = torch.cat(diag_seq, dim=1)   # (1, T, D)
        proc_seq = torch.cat(proc_seq, dim=1)
        drug_seq = torch.cat(drug_seq, dim=1)

        delta_t_seq = torch.stack(delta_t_seq).unsqueeze(0)  # (1, 1, T)

        # === 使用 decompose 进行分解 ===
        if self.decompose is not None:
            diag_seq = self.decompose(diag_seq, delta_t_seq)
            proc_seq = self.decompose(proc_seq, delta_t_seq)

        # === 模态编码 ===
        # HAHA
        # o_diag = self.transformer_encoder(diag_seq) # (1, T, D)
        # o_proc = self.transformer_encoder(proc_seq) # (1, T, D)

        # === 健康状况获取 ===
        # patient_representations = torch.cat([o_diag, o_proc,drug_seq],dim=-1).squeeze(dim=0) # (seq, dim*4)
        patient_representations = torch.cat([diag_seq, proc_seq,drug_seq],dim=-1).squeeze(dim=0) # (seq, dim*4)

        attn_output, attn_output_weights = self.attention(patient_representations, patient_representations,
                                                          patient_representations)

        # === 健康状况获取 ===
        # queries = self.query(attn_output)
        queries = self.query(patient_representations)

        return queries,delta_t_seq

class MELORA(nn.Module):
    def __init__(
            self,
            vocab_size,
            ehr_adj,
            ddi_adj,
            emb_dim,
            sigma,
            device=torch.device('cpu:0'),
            ddi_in_memory=True,
            fft_mode='full',
            use_time_decay=True,
            use_sim_decay=True,
            use_ddi_gcn=True,
    ):

        super(MELORA, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.4)

        self.ddi_in_memory = ddi_in_memory
        self.use_time_decay = use_time_decay
        self.use_sim_decay = use_sim_decay
        self.use_ddi_gcn = use_ddi_gcn

        self.patient_encoder = PatientEncoder(
            vocab_size=vocab_size,  # [诊断, 手术, 药物]
            emb_dim=emb_dim,
            device=device,
            fft_mode=fft_mode,
        )

        if self.use_sim_decay:
            self.similarity_function = SimilarityFunctions(sigma)
            self.sim_decay = SimilarityDecay(decay_rate=1.0)
        else:
            self.similarity_function = None
            self.sim_decay = None

        if self.use_time_decay:
            self.time_decay = TimeDecay(init_tau_years=2.0, time_scale=365.0).to(device)
        else:
            self.time_decay = None

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Linear(emb_dim * 3,  vocab_size[2])

        self.init_weights()

    def forward(self, input):
        # generate medical embeddings and queries
        '''I:generate current input'''
        queries_fused,delta_t_seq = self.patient_encoder(input)  # (seq_len, emb_dim)

        query = queries_fused[-1:].clone()

        decay_list = []

        query = queries_fused[-1:].clone()

        # graph memory module
        '''G:generate graph memory bank and insert history information'''
        if self.ddi_in_memory and self.use_ddi_gcn:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter
        else:
            drug_memory = self.ehr_gcn()

        if len(input) > 1:
            history_keys = queries_fused[:-1].clone()
            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1

            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)

        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)




