import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 130
DATA_DIR = '../data'
TRAIN_FILE = 'fold%s_train.json'
VALID_FILE = 'fold%s_valid.json'
TEST_FILE = 'fold%s_test.json'

# Storing all clauses containing sentimental word, based on the ANTUSD lexicon 'opinion_word_simplified.csv'. see https://academiasinicanlplab.github.io
SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses.pkl'


class Config(object):
    def __init__(self):
        self.split = 'split10'
        # self.split = 'split20'
        self.epochs = 40
        self.lr = 1e-5
        self.batch_size = 2
        self.gradient_accumulation_steps = 2
        self.warmup_proportion = 0.1
        self.K = 5
        self.dropout_hidden = 0.1
        self.dropout_pair = 0.1
        self.glm_hidden = 1024
        self.pos_emb_dim = 50
        self.max_sentence_length = 768
        self.emotion_enhanced = True
        self.use_rgcn = True
        self.pfn = True
