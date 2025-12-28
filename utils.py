import pandas as pd
import string
import re
import torch
import nltk
import hanlp
import pickle
import os
import json
from collections import Counter
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import math
from datetime import datetime
import csv
from tqdm import tqdm
import sacrebleu

# ==================== 1. 文本预处理与分词 ====================

_chinese_tokenizer = None

def get_chinese_tokenizer():
    global _chinese_tokenizer
    if _chinese_tokenizer is None:
        # 使用 HanLP 的精简版分词器
        _chinese_tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    return _chinese_tokenizer

def preprocess_english(text):
    text = text.lower()
    # 移除特殊标点
    special_punc = "''""–—…”“’‘" + string.punctuation
    translator = str.maketrans('', '', special_punc)
    text = text.translate(translator)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_chinese(text):
    chinese_punctuation = '，。！？、；：""''（）【】《》—…·￥""”“’‘' + string.punctuation
    text = re.sub(f'[{re.escape(chinese_punctuation)}]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_english(text):
    return nltk.word_tokenize(preprocess_english(str(text)))

def tokenize_chinese_hanlp(text):
    tokenizer = get_chinese_tokenizer()
    return tokenizer(preprocess_chinese(str(text)))

# ==================== 2. 词汇表管理 ====================

class Vocabulary:
    def __init__(self, name="vocab", min_freq=1):
        self.name = name
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()

    def build_vocab_from_tokens(self, all_tokens_list):
        for tokens in all_tokens_list:
            self.word_freq.update(tokens)
        
        curr_idx = len(self.word2idx)
        for word, freq in self.word_freq.most_common():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = curr_idx
                self.idx2word[curr_idx] = word
                curr_idx += 1
        print(f"[{self.name}] 词汇表构建完成: {len(self.word2idx)} 个词")

    def encode(self, tokens):
        return [self.word2idx.get(t, self.word2idx['<UNK>']) for t in tokens]

    def decode(self, indices, skip_special=True):
        tokens = []
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>')
            if skip_special and word in ['<PAD>', '<SOS>', '<EOS>']:
                continue
            tokens.append(word)
            if word == '<EOS>': break
        return tokens

    def save(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'word2idx': self.word2idx, 'min_freq': self.min_freq, 'name': self.name}, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = cls(name=data['name'], min_freq=data['min_freq'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(k): v for k, v in {v: k for k, v in vocab.word2idx.items()}.items()}
        return vocab
    
    def __len__(self):
        return len(self.word2idx)

# ==================== 3. 数据集与加载 ====================

class TranslationDataset(Dataset):
    def __init__(self, en_tokens, zh_tokens, en_vocab, zh_vocab):
        self.en_data = en_tokens
        self.zh_data = zh_tokens
        self.en_vocab = en_vocab
        self.zh_vocab = zh_vocab

    def __len__(self):
        return len(self.en_data)

    def __getitem__(self, idx):
        # 英文作为 Source (Encoder 输入), 中文作为 Target (Decoder 输入)
        en_ids = [self.en_vocab.word2idx['<SOS>']] + self.en_vocab.encode(self.en_data[idx]) + [self.en_vocab.word2idx['<EOS>']]
        zh_ids = [self.zh_vocab.word2idx['<SOS>']] + self.zh_vocab.encode(self.zh_data[idx]) + [self.zh_vocab.word2idx['<EOS>']]
        return torch.tensor(en_ids), torch.tensor(zh_ids)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=0)
    return src_padded, trg_padded

# ==================== 4. 评估指标 (BLEU) ====================


def compute_bleu(references: list, hypotheses: list):
    """
    使用 sacrebleu 计算语料库级别的 BLEU 分数
    
    Args:
        references: List[List[str]] - 真值分词列表
        hypotheses: List[List[str]] - 预测分词列表
    
    Returns:
        BLEU 分数 (0-100)
    """
    
    # sacrebleu 需要字符串格式
    hyp_sentences = [' '.join(tokens) for tokens in hypotheses]
    ref_sentences = [' '.join(tokens) for tokens in references]
    
    # 计算 BLEU
    bleu = sacrebleu.corpus_bleu(hyp_sentences, [ref_sentences])
    
    return bleu.score / 100  # 转换为 0-1 范围

# ==================== 5. 实验追踪与日志 ====================

def create_experiment_dir(base_dir='experiments', prefix='exp'):
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    exp_path = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(exp_path, exist_ok=True)
    return exp_path

def save_hyperparameters(exp_dir, config='config.json'):
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def init_csv_logger(exp_dir):
    path = os.path.join(exp_dir, 'log.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_bleu', 'is_loss_best', 'is_bleu_best'])
    return path

def log_epoch(path, data):
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([data['epoch'], data['train_loss'], data['val_loss'], data['val_bleu'], data['is_loss_best'], data['is_bleu_best']])

def save_test_results(exp_dir, loss, bleu, examples):
    res = {"test_loss": loss, "test_bleu": bleu, "examples": examples}
    with open(os.path.join(exp_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)

# ==================== 6. 数据加载全流程 (带 Cache) ====================

def load_and_prepare_data(data_path, cache_dir, en_min_freq=2, zh_min_freq=2):
    os.makedirs(cache_dir, exist_ok=True)
    token_cache = os.path.join(cache_dir, 'tokenized_data' + f'_en{en_min_freq}_zh{zh_min_freq}.pkl')
    
    if os.path.exists(token_cache):
        with open(token_cache, 'rb') as f:
            data = pickle.load(f)
        print("从缓存加载分词数据...")
    else:
        # 读取数据 (假设是 jsonl 格式)
        train_df = pd.read_json(os.path.join(data_path, 'train_100k.jsonl'), lines=True)
        valid_df = pd.read_json(os.path.join(data_path, 'valid.jsonl'), lines=True)
        test_df = pd.read_json(os.path.join(data_path, 'test.jsonl'), lines=True)
        
        data = {
            'train': {'en_tokens': [tokenize_english(s) for s in train_df['en']], 'zh_tokens': [tokenize_chinese_hanlp(s) for s in train_df['zh']]},
            'valid': {'en_tokens': [tokenize_english(s) for s in valid_df['en']], 'zh_tokens': [tokenize_chinese_hanlp(s) for s in valid_df['zh']]},
            'test': {'en_tokens': [tokenize_english(s) for s in test_df['en']], 'zh_tokens': [tokenize_chinese_hanlp(s) for s in test_df['zh']]}
        }

            
        # 构建词汇表 (仅基于训练集)
        en_vocab = Vocabulary("English", en_min_freq)
        en_vocab.build_vocab_from_tokens(data['train']['en_tokens'])
        zh_vocab = Vocabulary("Chinese", zh_min_freq)
        zh_vocab.build_vocab_from_tokens(data['train']['zh_tokens'])
        
        data['en_vocab'] = en_vocab
        data['zh_vocab'] = zh_vocab
        
        with open(token_cache, 'wb') as f:
            pickle.dump(data, f)

        
    return data