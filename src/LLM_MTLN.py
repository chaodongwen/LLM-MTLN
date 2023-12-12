import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from config import *
from utils.utils import *


class LLM_MTLN(nn.Module):
    def __init__(self, configs):
        super(LLM_MTLN, self).__init__()
        self.configs = configs
        self.dropout_hidden = nn.Dropout(configs.dropout_hidden)
        self.dropout_pair = nn.Dropout(configs.dropout_pair)
        self.glm = AutoModel.from_pretrained("glm-large-chinese", trust_remote_code=True)

        self.rgcn_network = RGCN_Network(configs)
        self.feature_extractor = Partition_Encoder(configs.glm_hidden, configs.dropout_hidden)
        self.pred = Pre_Predictions(configs)
        self.pair_construct = Pair_Construct(configs)
        self.pred_navie = Pre_Predictions_Navie(configs)
        self.pair_construct_navie = Pair_Construct_Navie(configs)

    def forward(self, glm_token_b, glm_segments_b, glm_masks_b, glm_clause_b, glm_clause_sep_b,
                doc_len_b, adj_b, emo_pos, cau_pos):
        glm_output = self.glm(input_ids=glm_token_b.to(DEVICE),
                              attention_mask=glm_masks_b.to(DEVICE),
                              token_type_ids=glm_segments_b.to(DEVICE))

        doc_sents_h = self.batched_index_select(glm_output, glm_clause_b.to(DEVICE))
        # 利用位置感知的RGCN网络对子句关系进行建模
        if self.configs.use_rgcn:
            doc_sents_h = self.rgcn_network(doc_sents_h)

        # 利用pfn网络，从原始子句特征中划分出情绪特征、原因特征和交互特征
        if self.configs.pfn:
            doc_sents_h = doc_sents_h.transpose(0, 1)
            if self.training:
                doc_sents_h = self.dropout_hidden(doc_sents_h)
            h_emo, h_cau, h_share = self.feature_extractor(doc_sents_h, layers=1)
            pred_e, pred_c, _, _ = self.pred(h_emo, h_cau, h_share)
            couples_pred, emo_cau_pos = self.pair_construct(h_emo, h_cau, h_share)
        else:
            pred_e, pred_c = self.pred_navie(doc_sents_h)
            couples_pred, emo_cau_pos = self.pair_construct_navie(doc_sents_h)

        return couples_pred, emo_cau_pos, pred_e, pred_c
    
    def batched_index_select(self, glm_output, glm_clause_b):
        hidden_state = glm_output[0]
        dummy = glm_clause_b.unsqueeze(2).expand(glm_clause_b.size(0), glm_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy)  # 选取每个句子的CLS向量
        return doc_sents_h

    def loss_rank(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        couples_true, couples_mask, doc_couples_pred = self.output_util(couples_pred, emo_cau_pos, doc_couples, y_mask, test)

        couples_mask = torch.BoolTensor(couples_mask).to(DEVICE)
        couples_true = torch.FloatTensor(couples_true).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        couples_true = couples_true.masked_select(couples_mask)
        couples_pred = couples_pred.masked_select(couples_mask)
        loss_couple = criterion(couples_pred, couples_true)

        return loss_couple, doc_couples_pred

    def output_util(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        batch, n_couple = couples_pred.size()
        couples_true, couples_mask = [], []
        doc_couples_pred = []
        for i in range(batch):
            y_mask_i = y_mask[i]
            max_doc_idx = sum(y_mask_i)

            doc_couples_i = doc_couples[i]
            couples_true_i = []
            couples_mask_i = []
            for couple_idx, emo_cau in enumerate(emo_cau_pos):
                if emo_cau[0] > max_doc_idx or emo_cau[1] > max_doc_idx:
                    couples_mask_i.append(0)
                    couples_true_i.append(0)
                else:
                    couples_mask_i.append(1)
                    couples_true_i.append(1 if emo_cau in doc_couples_i else 0)

            couples_pred_i = couples_pred[i]
            doc_couples_pred_i = []
            if test:
                if torch.sum(torch.isnan(couples_pred_i)) > 0:
                    k_idx = [0] * 3
                else:
                    _, k_idx = torch.topk(couples_pred_i, k=3, dim=0)
                # (位置，网络输出的得分) 相当于是取3个得分最高的二维坐标点
                doc_couples_pred_i = [(emo_cau_pos[idx], couples_pred_i[idx].tolist()) for idx in k_idx]  # [([7, 7], -2.118), ([9, 7], -2.168), ([6, 7], -2.194)]

            couples_true.append(couples_true_i)
            couples_mask.append(couples_mask_i)
            doc_couples_pred.append(doc_couples_pred_i)
        return couples_true, couples_mask, doc_couples_pred

    def loss_pre(self, pred_e, pred_c, y_emotions, y_causes, y_mask):
        y_mask = torch.BoolTensor(y_mask).to(DEVICE)
        y_emotions = torch.FloatTensor(y_emotions).to(DEVICE)
        y_causes = torch.FloatTensor(y_causes).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

        pred_e = pred_e.masked_select(y_mask)
        true_e = y_emotions.masked_select(y_mask)
        loss_e = criterion(pred_e, true_e)

        pred_c = pred_c.masked_select(y_mask)
        true_c = y_causes.masked_select(y_mask)
        loss_c = criterion(pred_c, true_c)

        return loss_e, loss_c


class RGCN_Network(nn.Module):
    def __init__(self, configs):
        super(RGCN_Network, self).__init__()
        input_dim = configs.glm_hidden
        output_dim = configs.glm_hidden
        self.K = configs.K
        self.rel_num = self.K + 1
        self.rgcn = RGCNConv(input_dim, output_dim, self.rel_num, num_bases=2)

    def forward(self, x):
        batch_size = x.shape[0]
        slen = x.shape[1]
        x_dim = x.shape[2]
        # 创建限定邻接矩阵，target_index是情绪子句的索引，source_index是原因子句的索引
        thr = self.K
        single_relative_position_list = []
        for target_index in range(slen):
            temp_relative_position = [thr] * slen
            for source_index in range(slen):
                if abs(source_index + 1 - target_index) <= thr:  # cause_index - emotion_index
                    temp_relative_position[source_index] = abs(source_index + 1 - target_index)
                else:
                    temp_relative_position[source_index] = thr
            single_relative_position_list.append(temp_relative_position)
        rel_adj = torch.tensor(single_relative_position_list, dtype=torch.long).to(DEVICE)
        edge_type = torch.flatten(rel_adj).long().to(DEVICE)

        # 创建索引
        base_idx = np.arange(slen)
        start = np.concatenate([base_idx.reshape(-1, 1)] * slen, axis=1).reshape(1, -1)[0]
        end = np.concatenate([base_idx] * slen, axis=0)
        index = [start, end]
        index = torch.tensor(index).long().to(DEVICE)

        # 利用关系图卷积网络对文档特征进行处理
        out = self.rgcn(x[0], index, edge_type).unsqueeze(0)
        for i in range(1, batch_size):
            h = self.rgcn(x[i], index, edge_type)
            out = torch.cat((out, h.unsqueeze(0)), dim=0)

        return out


def cumsoftmax(x):
    return torch.cumsum(F.softmax(x,-1),dim=-1)


class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.bool
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout), self.bias)


class pfn_unit(nn.Module):
    def __init__(self, input_size, drop=0.1):
        super(pfn_unit, self).__init__()

        self.hidden_transform = LinearDropConnect(300, 5 * 300, bias=True, dropout=drop)
        self.input_transform = nn.Linear(input_size, 5 * 300, bias=True)

        self.transform = nn.Linear(300 * 3, 300)
        self.drop_weight_modules = [self.hidden_transform]

        torch.nn.init.orthogonal_(self.input_transform.weight, gain=1)
        torch.nn.init.orthogonal_(self.transform.weight, gain=1)

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()

    def forward(self, x, hidden):
        h_in, c_in = hidden

        gates = self.input_transform(x) + self.hidden_transform(h_in)
        c, eg_cin, rg_cin, eg_c, rg_c = gates[:, :].chunk(5, 1)

        eg_cin = 1 - cumsoftmax(eg_cin)
        rg_cin = cumsoftmax(rg_cin)

        eg_c = 1 - cumsoftmax(eg_c)
        rg_c = cumsoftmax(rg_c)

        c = torch.tanh(c)

        overlap_c = rg_c * eg_c
        upper_c = rg_c - overlap_c
        downer_c = eg_c - overlap_c

        overlap_cin = rg_cin * eg_cin
        upper_cin = rg_cin - overlap_cin
        downer_cin = eg_cin - overlap_cin

        share = overlap_cin * c_in + overlap_c * c

        c_cau = upper_cin * c_in + upper_c * c
        c_emo = downer_cin * c_in + downer_c * c
        c_share = share

        h_cau = torch.tanh(c_cau)
        h_emo = torch.tanh(c_emo)
        h_share = torch.tanh(c_share)

        c_out = torch.cat((c_cau, c_emo, c_share), dim=-1)
        c_out_2 = self.transform(c_out)
        h_out = torch.tanh(c_out_2)

        return (h_out, c_out_2), (h_emo, h_cau, h_share), c_out_2


class Partition_Encoder(nn.Module):
    def __init__(self, input_size, drop=0.1):
        super(Partition_Encoder, self).__init__()
        self.unit = pfn_unit(input_size, drop)  # input_size=768

    def hidden_init(self, batch_size):
        h0 = torch.zeros(batch_size, 300).requires_grad_(False).to(DEVICE)
        c0 = torch.zeros(batch_size, 300).requires_grad_(False).to(DEVICE)
        return (h0, c0)

    def forward(self, x, layers=1):
        seq_len = x.size(0)
        batch_size = x.size(1)
        h_emo, h_cau, h_share = [], [], []
        if self.training:
            self.unit.sample_masks()  # 对权重进行归一化

        for layer in range(layers):
            hidden = self.hidden_init(batch_size)
            output = []
            for t in range(seq_len):
                # (h_out, c_out_2), (h_emo, h_cau, h_share), c_out_2
                hidden, h_task, output_layer = self.unit(x[t, :, :], hidden)
                output.append(output_layer)
                if layer == layers - 1:
                    h_emo.append(h_task[0])
                    h_cau.append(h_task[1])
                    h_share.append(h_task[2])

            x = torch.stack(output, dim=0)

        h_emo = torch.stack(h_emo, dim=0)
        h_cau = torch.stack(h_cau, dim=0)
        h_share = torch.stack(h_share, dim=0)

        return h_emo.transpose(0, 1), h_cau.transpose(0, 1), h_share.transpose(0, 1)  # [batch, seq_len, hidden]


class Pair_Construct(nn.Module):
    def __init__(self, configs):
        super(Pair_Construct, self).__init__()
        self.hidden_size = 300

        self.hid2hid = nn.Linear(self.hidden_size * 2 + configs.pos_emb_dim, self.hidden_size)
        self.hid2rel = nn.Linear(self.hidden_size, 1)
        self.ln = nn.LayerNorm(self.hidden_size)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(configs.dropout_pair)

        self.K = configs.K
        self.pos_layer = nn.Embedding(2 * self.K + 1, configs.pos_emb_dim)
        nn.init.xavier_uniform_(self.pos_layer.weight)

        torch.nn.init.orthogonal_(self.hid2hid.weight, gain=1)
        torch.nn.init.orthogonal_(self.hid2rel.weight, gain=1)

    def forward(self, h_e, h_c, h_share):
        batch_size, length, hidden = h_c.size()
        h_e = h_e + h_share
        h_c = h_c + h_share
        couples, rel_pos, emo_cau_pos = self.couple_generator(h_e, h_c, self.K)

        # 构造相对位置矩阵
        rel_pos = rel_pos + self.K
        rel_pos_emb = self.pos_layer(rel_pos)
        kernel = self.kernel_generator(rel_pos)
        kernel = kernel.unsqueeze(0).expand(batch_size, -1, -1)
        rel_pos_emb = torch.matmul(kernel, rel_pos_emb)
        # 融入相对位置信息
        pair = torch.cat((couples, rel_pos_emb), dim=-1)
        # 对pair进行MLP处理
        pair = self.ln(self.hid2hid(pair))
        pair = self.elu(self.dropout(pair))
        pair = self.hid2rel(pair)
        pair = pair.squeeze(-1)

        return pair, emo_cau_pos

    def couple_generator(self, H_e, H_c, K):
        batch, seq_len, feat_dim = H_e.size()
        P_left = torch.cat([H_e] * seq_len, dim=2)
        P_left = P_left.reshape(-1, seq_len * seq_len, feat_dim)
        P_right = torch.cat([H_c] * seq_len, dim=1)
        P = torch.cat([P_left, P_right], dim=2)

        base_idx = np.arange(1, seq_len + 1)
        emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
        cau_pos = np.concatenate([base_idx] * seq_len, axis=0)

        rel_pos = cau_pos - emo_pos
        rel_pos = torch.LongTensor(rel_pos).to(DEVICE)
        emo_pos = torch.LongTensor(emo_pos).to(DEVICE)
        cau_pos = torch.LongTensor(cau_pos).to(DEVICE)

        if seq_len > K + 1:
            rel_mask = np.array(list(map(lambda x: -K <= x <= K, rel_pos.tolist())), dtype=np.int)
            rel_mask = torch.BoolTensor(rel_mask).to(DEVICE)
            rel_pos = rel_pos.masked_select(rel_mask)
            emo_pos = emo_pos.masked_select(rel_mask)
            cau_pos = cau_pos.masked_select(rel_mask)

            rel_mask = rel_mask.unsqueeze(1).expand(-1, 2 * feat_dim)
            rel_mask = rel_mask.unsqueeze(0).expand(batch, -1, -1)

            P = P.masked_select(rel_mask)
            P = P.reshape(batch, -1, 2 * feat_dim)

        assert rel_pos.size(0) == P.size(1)
        rel_pos = rel_pos.unsqueeze(0).expand(batch, -1)

        emo_cau_pos = []
        for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
            emo_cau_pos.append([emo, cau])  # 该窗口k下，可能出现的所有情绪原因对
        return P, rel_pos, emo_cau_pos

    def kernel_generator(self, rel_pos):
        n_couple = rel_pos.size(1)
        rel_pos_ = rel_pos[0].type(torch.FloatTensor).to(DEVICE)
        kernel_left = torch.cat([rel_pos_.reshape(-1, 1)] * n_couple, dim=1)
        kernel = kernel_left - kernel_left.transpose(0, 1)
        return torch.exp(-(torch.pow(kernel, 2)))


class Pair_Construct_Navie(nn.Module):
    def __init__(self, configs):
        super(Pair_Construct_Navie, self).__init__()
        self.hid2hid = nn.Linear(configs.glm_hidden * 2 + configs.pos_emb_dim, configs.glm_hidden)
        self.hid2rel = nn.Linear(configs.glm_hidden, 1)
        self.ln = nn.LayerNorm(configs.glm_hidden)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(configs.dropout_pair)

        self.K = configs.K
        self.pos_layer = nn.Embedding(2 * self.K + 1, configs.pos_emb_dim)
        nn.init.xavier_uniform_(self.pos_layer.weight)

        torch.nn.init.orthogonal_(self.hid2hid.weight, gain=1)
        torch.nn.init.orthogonal_(self.hid2rel.weight, gain=1)

    def forward(self, doc_sents_h):
        batch_size, _, _ = doc_sents_h.size()
        couples, rel_pos, emo_cau_pos = self.couple_generator(doc_sents_h, self.K)

        rel_pos = rel_pos + self.K
        rel_pos_emb = self.pos_layer(rel_pos)
        kernel = self.kernel_generator(rel_pos)
        kernel = kernel.unsqueeze(0).expand(batch_size, -1, -1)
        rel_pos_emb = torch.matmul(kernel, rel_pos_emb)
        # 融入相对位置信息
        pair = torch.cat([couples, rel_pos_emb], dim=2)
        # 对pair进行MLP处理
        pair = self.ln(self.hid2hid(pair))
        pair = self.elu(self.dropout(pair))
        pair = self.hid2rel(pair)
        pair = pair.squeeze(-1)

        return pair, emo_cau_pos

    def couple_generator(self, H, K):
        batch, seq_len, feat_dim = H.size()
        P_left = torch.cat([H] * seq_len, dim=2)
        P_left = P_left.reshape(-1, seq_len * seq_len, feat_dim)
        P_right = torch.cat([H] * seq_len, dim=1)
        P = torch.cat([P_left, P_right], dim=2)

        base_idx = np.arange(1, seq_len + 1)
        emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
        cau_pos = np.concatenate([base_idx] * seq_len, axis=0)

        rel_pos = cau_pos - emo_pos
        rel_pos = torch.LongTensor(rel_pos).to(DEVICE)
        emo_pos = torch.LongTensor(emo_pos).to(DEVICE)
        cau_pos = torch.LongTensor(cau_pos).to(DEVICE)

        if seq_len > K + 1:
            rel_mask = np.array(list(map(lambda x: -K <= x <= K, rel_pos.tolist())), dtype=np.int)
            rel_mask = torch.BoolTensor(rel_mask).to(DEVICE)
            rel_pos = rel_pos.masked_select(rel_mask)
            emo_pos = emo_pos.masked_select(rel_mask)
            cau_pos = cau_pos.masked_select(rel_mask)

            rel_mask = rel_mask.unsqueeze(1).expand(-1, 2 * feat_dim)
            rel_mask = rel_mask.unsqueeze(0).expand(batch, -1, -1)
            P = P.masked_select(rel_mask)
            P = P.reshape(batch, -1, 2 * feat_dim)
        assert rel_pos.size(0) == P.size(1)
        rel_pos = rel_pos.unsqueeze(0).expand(batch, -1)

        emo_cau_pos = []
        for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
            emo_cau_pos.append([emo, cau])
        return P, rel_pos, emo_cau_pos

    def kernel_generator(self, rel_pos):
        n_couple = rel_pos.size(1)
        rel_pos_ = rel_pos[0].type(torch.FloatTensor).to(DEVICE)
        kernel_left = torch.cat([rel_pos_.reshape(-1, 1)] * n_couple, dim=1)
        kernel = kernel_left - kernel_left.transpose(0, 1)
        return torch.exp(-(torch.pow(kernel, 2)))


class Pre_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions, self).__init__()
        self.line_e = nn.Linear(300 * 2, 300)
        self.out_e_subwork = nn.Linear(300, 1)
        self.line_c = nn.Linear(300 * 2, 300)
        self.out_c_subwork = nn.Linear(300, 1)

        self.leakyrule = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.orthogonal_(self.line_e.weight, gain=1)
        torch.nn.init.orthogonal_(self.out_e_subwork.weight, gain=1)
        torch.nn.init.orthogonal_(self.line_c.weight, gain=1)
        torch.nn.init.orthogonal_(self.out_c_subwork.weight, gain=1)

    def forward(self, sent_e, sent_c, share):
        sent_e_subwork = self.leakyrule(self.line_e(torch.cat((sent_e, share), -1)))
        pred_e_subwork = self.out_e_subwork(sent_e_subwork)
        sent_c_subwork = self.leakyrule(self.line_c(torch.cat((sent_c, share), -1)))
        pred_c_subwork = self.out_c_subwork(sent_c_subwork)

        pred_e_subwork = pred_e_subwork.squeeze(2)
        pred_c_subwork = pred_c_subwork.squeeze(2)

        return pred_e_subwork, pred_c_subwork, sent_e_subwork, sent_c_subwork


class Pre_Predictions_Navie(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions_Navie, self).__init__()
        self.out_e = nn.Linear(configs.glm_hidden, 1)
        self.out_c = nn.Linear(configs.glm_hidden, 1)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h)
        pred_c = self.out_c(doc_sents_h)
        return pred_e.squeeze(2), pred_c.squeeze(2)
