import sys
sys.path.append('..')
import os
from os.path import join
import scipy.sparse as sp
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
from config import *
from utils.utils import *


def build_train_data(configs, fold_id, shuffle=True):
    train_dataset = MyDataset(configs, fold_id, data_type='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=shuffle, collate_fn=glm_batch_preprocessing)
    return train_loader


def build_inference_data(configs, fold_id, data_type):
    dataset = MyDataset(configs, fold_id, data_type)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=configs.batch_size,
                                              shuffle=False, collate_fn=glm_batch_preprocessing)
    return data_loader


class MyDataset(Dataset):
    def __init__(self, configs, fold_id, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.split = configs.split
        self.emotion_enhanced = configs.emotion_enhanced
        self.glm_tokenizer = AutoTokenizer.from_pretrained("glm-large-chinese", trust_remote_code=True)

        self.data_type = data_type
        self.train_file = join(data_dir, self.split, TRAIN_FILE % fold_id)
        self.valid_file = join(data_dir, self.split, VALID_FILE % fold_id)
        self.test_file = join(data_dir, self.split, TEST_FILE % fold_id)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs
        self.K = configs.K
        self.max_sentence_length = configs.max_sentence_length

        self.doc_couples_list, self.y_emotions_list, self.y_causes_list, self.doc_len_list, self.doc_id_list, \
        self.glm_token_idx_list, self.glm_clause_idx_list, self.glm_clause_sep_idx_list, self.glm_segments_idx_list, \
        self.glm_token_lens_list, self.emo_pos_list, self.cau_pos_list = self.read_data_file(self.data_type)

    def __len__(self):
        return len(self.y_emotions_list)

    def __getitem__(self, idx):
        doc_couples, y_emotions, y_causes = self.doc_couples_list[idx], self.y_emotions_list[idx], self.y_causes_list[idx]
        doc_len, doc_id = self.doc_len_list[idx], self.doc_id_list[idx]
        glm_token_idx, glm_clause_idx = self.glm_token_idx_list[idx], self.glm_clause_idx_list[idx]
        glm_clause_sep_idx, glm_segments_idx = self.glm_clause_sep_idx_list[idx], self.glm_segments_idx_list[idx]
        glm_token_lens, emo_pos, cau_pos = self.glm_token_lens_list[idx], self.emo_pos_list[idx], self.cau_pos_list[idx]

        glm_token_idx = torch.LongTensor(glm_token_idx)
        glm_clause_idx = torch.LongTensor(glm_clause_idx)
        glm_clause_sep_idx = torch.LongTensor(glm_clause_sep_idx)
        glm_segments_idx = torch.LongTensor(glm_segments_idx)

        assert doc_len == len(y_emotions)
        return doc_couples, y_emotions, y_causes, doc_len, doc_id, \
               glm_token_idx, glm_clause_idx, glm_clause_sep_idx, \
               glm_segments_idx, glm_token_lens, emo_pos, cau_pos

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        doc_id_list = []
        doc_len_list = []
        doc_couples_list = []
        y_emotions_list, y_causes_list = [], []
        glm_token_idx_list = []
        glm_clause_idx_list = []
        glm_clause_sep_idx_list = []
        glm_segments_idx_list = []
        glm_token_lens_list = []
        emo_pos_list, cau_pos_list = [], []

        data_list = read_json(data_file)
        # SENTIMENTAL_CLAUSE_DICT是通过ANTUSD情感词典对文档子句进行分析得到，从文档中抽取出了带有情感的句子（并未利用原始数据中的标签信息）
        emotional_clauses = read_b(os.path.join(DATA_DIR, SENTIMENTAL_CLAUSE_DICT))
        for doc in data_list:
            doc_id = doc['doc_id']
            doc_len = doc['doc_len']
            doc_couples = doc['pairs']
            doc_emotions, doc_causes = zip(*doc_couples)
            doc_couples = list(map(lambda x: list(x), doc_couples))
            y_emotions, y_causes = [], []
            emotional_clauses_i = emotional_clauses[doc_id]
            doc_clauses = doc['clauses']
            doc_str = ''
            for i in range(doc_len):
                emotion_label = int(i + 1 in doc_emotions)
                cause_label = int(i + 1 in doc_causes)
                y_emotions.append(emotion_label)
                y_causes.append(cause_label)
                clause = doc_clauses[i]
                clause_id = clause['clause_id']
                assert int(clause_id) == i + 1
                if self.emotion_enhanced:
                    if int(clause_id) in emotional_clauses_i:
                        doc_str += '[CLS] ' + '[/e] ' + clause['clause'] + ' [/e] ' + '[SEP] '
                    else:
                        doc_str += '[CLS] ' + clause['clause'] + ' [SEP] '
                else:
                    doc_str += '[CLS] ' + clause['clause'] + ' [SEP] '
            # [SEP]=50001  [CLS]=50002  [MASK]=50003
            indexed_tokens = self.glm_tokenizer.encode(doc_str.strip(), add_special_tokens=False)
            clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 50002]
            clause_sep_indices = [i for i, x in enumerate(indexed_tokens) if x == 50001]
            doc_token_len = len(indexed_tokens)
            assert doc_token_len <= self.max_sentence_length

            segments_ids = []
            segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 50002]
            segments_indices.append(len(indexed_tokens))
            for i in range(len(segments_indices) - 1):
                semgent_len = segments_indices[i + 1] - segments_indices[i]
                if i % 2 == 0:
                    segments_ids.extend([0] * semgent_len)
                else:
                    segments_ids.extend([1] * semgent_len)

            assert len(clause_indices) == len(clause_sep_indices) == doc_len
            assert len(segments_ids) == len(indexed_tokens) == doc_token_len

            glm_token_idx_list.append(indexed_tokens)  # 整个文档所有子句连起来后的token表示
            glm_clause_idx_list.append(clause_indices)  # 每个子句开始标识的index
            glm_clause_sep_idx_list.append(clause_sep_indices)  # 每个子句结束标识的index
            glm_segments_idx_list.append(segments_ids)  # 用0和1来区分不同的子句
            glm_token_lens_list.append(doc_token_len)  # 文档中所有子句的token的个数
            # 构造相对位置
            emo_pos, cau_pos = construct_relative_pos(doc_len, self.K)

            doc_couples_list.append(doc_couples)
            y_emotions_list.append(y_emotions)
            y_causes_list.append(y_causes)
            doc_len_list.append(doc_len)
            doc_id_list.append(doc_id)
            emo_pos_list.append(emo_pos)  # 情绪位置列表，例如[0,0,0,1,1,1,2,2,2]
            cau_pos_list.append(cau_pos)  # 原因位置列表，例如[0,1,2,0,1,2,0,1,2]

        return doc_couples_list, y_emotions_list, y_causes_list, doc_len_list, doc_id_list, \
               glm_token_idx_list, glm_clause_idx_list, glm_clause_sep_idx_list, glm_segments_idx_list, \
               glm_token_lens_list, emo_pos_list, cau_pos_list


def glm_batch_preprocessing(batch):
    doc_couples_b, y_emotions_b, y_causes_b, doc_len_b, doc_id_b, \
    glm_token_b, glm_clause_b, glm_clause_sep_b, \
    glm_segments_b, glm_token_lens_b, emo_pos_b, cau_pos_b = zip(*batch)

    y_mask_b, y_emotions_b, y_causes_b = pad_docs(doc_len_b, y_emotions_b, y_causes_b)
    adj_b = pad_matrices(doc_len_b)
    glm_token_b = pad_sequence(glm_token_b, batch_first=True, padding_value=0)
    glm_clause_b = pad_sequence(glm_clause_b, batch_first=True, padding_value=0)
    glm_clause_sep_b = pad_sequence(glm_clause_sep_b, batch_first=True, padding_value=0)
    glm_segments_b = pad_sequence(glm_segments_b, batch_first=True, padding_value=0)

    bsz, max_len = glm_token_b.size()
    glm_masks_b = np.zeros([bsz, max_len], dtype=np.float)
    for index, seq_len in enumerate(glm_token_lens_b):
        glm_masks_b[index][:seq_len] = 1

    glm_masks_b = torch.FloatTensor(glm_masks_b)
    assert glm_segments_b.shape == glm_token_b.shape
    assert glm_segments_b.shape == glm_masks_b.shape

    return np.array(doc_len_b), np.array(adj_b), np.array(y_emotions_b), np.array(y_causes_b), np.array(y_mask_b), \
           doc_couples_b, doc_id_b, glm_token_b, glm_segments_b, glm_masks_b, glm_clause_b, glm_clause_sep_b, \
           list(emo_pos_b), list(cau_pos_b)


def pad_docs(doc_len_b, y_emotions_b, y_causes_b):
    max_doc_len = max(doc_len_b)

    y_mask_b, y_emotions_b_, y_causes_b_ = [], [], []
    for y_emotions, y_causes in zip(y_emotions_b, y_causes_b):
        y_emotions_ = pad_list(y_emotions, max_doc_len, -1)
        y_causes_ = pad_list(y_causes, max_doc_len, -1)
        y_mask = list(map(lambda x: 0 if x == -1 else 1, y_emotions_))

        y_mask_b.append(y_mask)
        y_emotions_b_.append(y_emotions_)
        y_causes_b_.append(y_causes_)

    return y_mask_b, y_emotions_b_, y_causes_b_


def pad_matrices(doc_len_b):
    N = max(doc_len_b)
    adj_b = []
    for doc_len in doc_len_b:
        adj = np.ones((doc_len, doc_len))
        adj = sp.coo_matrix(adj)
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                            shape=(N, N), dtype=np.float32)
        adj_b.append(adj.toarray())
    return adj_b


def pad_list(element_list, max_len, pad_mark):
    element_list_pad = element_list[:]
    pad_mark_list = [pad_mark] * (max_len - len(element_list))
    element_list_pad.extend(pad_mark_list)
    return element_list_pad


def construct_relative_pos(seq_len, k):
    base_idx = np.arange(0, seq_len)
    emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
    cau_pos = np.concatenate([base_idx] * seq_len, axis=0)
    rel_pos = cau_pos - emo_pos
    rel_pos = torch.LongTensor(rel_pos).to(DEVICE)
    emo_pos = torch.LongTensor(emo_pos).to(DEVICE)
    cau_pos = torch.LongTensor(cau_pos).to(DEVICE)
    if seq_len > k + 1:
        rel_mask = np.array(list(map(lambda x: -k <= x <= k, rel_pos.tolist())), dtype=np.int)
        rel_mask = torch.BoolTensor(rel_mask).to(DEVICE)
        rel_pos = rel_pos.masked_select(rel_mask)
        emo_pos = emo_pos.masked_select(rel_mask)
        cau_pos = cau_pos.masked_select(rel_mask)

    return emo_pos, cau_pos
