import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution
import math


class FrequencyDecomposition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, delta_t=None):
        B, L, D = x.shape

        if L <= 1:
            return x, torch.zeros_like(x)

        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        f = x_fft.size(1)

        if delta_t is not None and L > 1:
            t = delta_t.view(-1)
            if t.numel() > 1:
                interval = t[1:] - t[:-1]
                interval = torch.abs(interval).mean()
                ratio = torch.sigmoid(1.0 / (interval + 1e-6))
                c = int((ratio * f).item())
            else:
                c = max(1, f // 2)
        else:
            c = max(1, f // 2)

        c = max(1, min(c, f - 1))

        low_fft = torch.zeros_like(x_fft)
        high_fft = torch.zeros_like(x_fft)

        low_fft[:, :c, :] = x_fft[:, :c, :]
        high_fft[:, c:, :] = x_fft[:, c:, :]

        low = torch.fft.irfft(low_fft, n=L, dim=1, norm='ortho')
        high = torch.fft.irfft(high_fft, n=L, dim=1, norm='ortho')

        return low, high


class TimeWeight(nn.Module):
    def __init__(self, init_lambda=0.01):
        super().__init__()
        self.lambda_raw = nn.Parameter(torch.tensor(float(init_lambda)))

    def forward(self, delta_t):
        delta_t = delta_t.view(1, -1, 1)
        current_t = delta_t[:, -1:, :]
        rel_t = torch.clamp(current_t - delta_t, min=0.0)
        lam = F.softplus(self.lambda_raw)
        return torch.exp(-lam * rel_t)


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

        self.emb_diag = nn.Embedding(vocab_size[0], emb_dim)
        self.emb_proc = nn.Embedding(vocab_size[1], emb_dim)
        self.dropout = nn.Dropout(0.4)

        self.decompose = FrequencyDecomposition()

        self.enc_sta = nn.GRU(
            input_size=emb_dim * 2,
            hidden_size=emb_dim,
            batch_first=True,
            bidirectional=True
        )

        self.enc_dyn = nn.GRU(
            input_size=emb_dim * 2,
            hidden_size=emb_dim,
            batch_first=True,
            bidirectional=True
        )

        self.time_weight = TimeWeight(init_lambda=0.01)
        self.gate = nn.Linear(2, 1)

    def mean_emb(self, emb_layer, idxs):
        if len(idxs) == 0:
            return torch.zeros((1, 1, self.emb_dim), device=self.device)
        idxs = torch.LongTensor(idxs).to(self.device)
        x = self.dropout(emb_layer(idxs))
        return x.mean(dim=0, keepdim=True).unsqueeze(0)

    def forward(self, input):
        diag_seq, proc_seq, delta_t_seq = [], [], []

        for adm in input:
            diag_seq.append(self.mean_emb(self.emb_diag, adm[0]))
            proc_seq.append(self.mean_emb(self.emb_proc, adm[1]))
            delta_t_seq.append(torch.tensor([adm[-1]], dtype=torch.float, device=self.device))

        diag_seq = torch.cat(diag_seq, dim=1)
        proc_seq = torch.cat(proc_seq, dim=1)
        delta_t_seq = torch.stack(delta_t_seq).view(1, -1, 1)

        if self.fft_mode == 'none':
            diag_low, diag_high = diag_seq, torch.zeros_like(diag_seq)
            proc_low, proc_high = proc_seq, torch.zeros_like(proc_seq)
        else:
            diag_low, diag_high = self.decompose(diag_seq, delta_t_seq)
            proc_low, proc_high = self.decompose(proc_seq, delta_t_seq)

            if self.fft_mode == 'low':
                diag_high = torch.zeros_like(diag_high)
                proc_high = torch.zeros_like(proc_high)
            elif self.fft_mode == 'high':
                diag_low = torch.zeros_like(diag_low)
                proc_low = torch.zeros_like(proc_low)

        low_input = torch.cat([diag_low, proc_low], dim=-1)
        high_input = torch.cat([diag_high, proc_high], dim=-1)

        h_lfc, _ = self.enc_sta(low_input)
        h_hfc, _ = self.enc_dyn(high_input)

        w_time = self.time_weight(delta_t_seq)

        sim = F.cosine_similarity(h_hfc, h_lfc, dim=-1).unsqueeze(-1)
        w_sim = F.softmax(sim, dim=1)

        gate_input = torch.cat([w_time, w_sim], dim=-1)
        gamma = torch.sigmoid(self.gate(gate_input))
        w = gamma * w_time + (1.0 - gamma) * w_sim

        h_hfc_refined = w * h_hfc
        q = torch.cat([h_lfc, h_hfc_refined], dim=-1)

        return q, h_lfc, h_hfc_refined, delta_t_seq


class ScaleMedicationAttention(nn.Module):
    def __init__(self, patient_dim, drug_dim):
        super().__init__()
        self.w_q = nn.Linear(patient_dim, drug_dim)
        self.w_k = nn.Linear(drug_dim, drug_dim)
        self.w_v = nn.Linear(drug_dim, drug_dim)

    def forward(self, patient_repr, medication_repr):
        q = self.w_q(patient_repr)
        k = self.w_k(medication_repr)
        v = self.w_v(medication_repr)
        score = torch.matmul(q, k.t()) / math.sqrt(k.size(-1))
        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, v)
        return out


class FAMED(nn.Module):
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

        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.use_ddi_gcn = use_ddi_gcn
        self.emb_dim = emb_dim

        self.patient_encoder = PatientEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            device=device,
            fft_mode=fft_mode,
        )

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)

        self.hist_med_embedding = nn.Embedding(vocab_size[2], emb_dim)
        self.alpha1 = nn.Parameter(torch.tensor(0.3))

        self.attn_hfc = ScaleMedicationAttention(patient_dim=emb_dim * 2, drug_dim=emb_dim)
        self.attn_lfc = ScaleMedicationAttention(patient_dim=emb_dim * 2, drug_dim=emb_dim)

        self.output = nn.Linear(emb_dim * 6, vocab_size[2])

        self.init_weights()

    def build_history_medication_embedding(self, input):
        hist_drugs = []
        for adm in input[:-1]:
            hist_drugs.extend(adm[2])

        e_m = torch.zeros(self.vocab_size[2], self.emb_dim, device=self.device)

        if len(hist_drugs) == 0:
            return e_m

        hist_drugs = torch.LongTensor(list(set(hist_drugs))).to(self.device)
        e_m[hist_drugs] = self.hist_med_embedding(hist_drugs)
        return e_m

    def forward(self, input):
        q, h_lfc, h_hfc, delta_t_seq = self.patient_encoder(input)

        q_t = q[:, -1, :]
        h_lfc_t = h_lfc[:, -1, :]
        h_hfc_t = h_hfc[:, -1, :]

        m_e = self.ehr_gcn()

        if self.ddi_in_memory and self.use_ddi_gcn:
            m_d = self.ddi_gcn()
            p_i = m_e - self.alpha1 * m_d + self.build_history_medication_embedding(input)
        else:
            p_i = m_e + self.build_history_medication_embedding(input)

        o_hfc = self.attn_hfc(h_hfc_t, p_i)
        o_lfc = self.attn_lfc(h_lfc_t, p_i)

        output = self.output(torch.cat([o_hfc, o_lfc, q_t], dim=-1))

        if self.training:
            pred_prob = torch.sigmoid(output)
            neg_pred_prob = pred_prob.t() * pred_prob
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()
            return output, batch_neg
        else:
            return output

    def init_weights(self):
        initrange = 0.1
        self.hist_med_embedding.weight.data.uniform_(-initrange, initrange)
        if isinstance(self.alpha1, nn.Parameter):
            self.alpha1.data.fill_(0.3)
