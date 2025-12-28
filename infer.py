import torch
import argparse
import os
import json

# 导入你的逻辑
from utils import (
    load_and_prepare_data,
    tokenize_chinese_hanlp
)
from transformer import build_model as build_transformer
from GRU import build_model as build_rnn

# ==================== 1. 模型加载逻辑 ====================
def load_model(args, device):
    print(f"--- 正在初始化 [{args.arch.upper()}] 翻译系统 ---")
    
    # 加载词表缓存
    data = load_and_prepare_data(
        args.data_path, args.cache_dir,
        en_min_freq=args.frequency, zh_min_freq=args.frequency
    )
    zh_vocab, en_vocab = data['zh_vocab'], data['en_vocab']

    if args.arch == 'transformer':
        # 基于截图 2 的参数: absolute, rmsnorm, d256, L3, h8, ff2048
        model = build_transformer(
            src_vocab_size=len(zh_vocab),
            trg_vocab_size=len(en_vocab),
            device=device,
            d_model=256,
            num_heads=8,
            num_layers=3,
            d_ff=2048,
            pos_encoding='absolute',
            norm_type='rmsnorm',
            max_len=args.max_len
        )
    else:
        # 基于截图 1 的参数: gru, multiplicative, emb300, hidden512
        model = build_rnn(
            src_vocab_size=len(zh_vocab),
            trg_vocab_size=len(en_vocab),
            device=device,
            emb_dim=300,
            hidden_dim=512,
            rnn_type='gru',
            attention_type='multiplicative'
        )

    print(f"加载权重: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict)
    model.eval()
    
    return model, zh_vocab, en_vocab

# ==================== 2. 交互式翻译界面 ====================
def start_inference(model, zh_vocab, en_vocab, device, args):
    print("\n" + "="*60)
    print(f" 🚀 统一翻译平台 | 当前架构: {args.arch.upper()}")
    print(f" 默认模式: {args.decode_method} (Beam Size: {args.beam_size})")
    print("="*60)
    print(" 指令: 'q' 退出 | 'greedy' 切换贪婪搜索 | 'beam N' 切换束搜索")

    cur_method = args.decode_method
    cur_beam = args.beam_size

    while True:
        try:
            text = input("\n[中文] >>> ").strip()
            if text.lower() in ['q', 'exit']: break
            
            # 实时切换解码参数
            if text.lower() == 'greedy':
                cur_method = 'greedy'; print("已切换至 Greedy Search"); continue
            if text.lower().startswith('beam'):
                try:
                    cur_beam = int(text.split()[1])
                    cur_method = 'beam'; print(f"已切换至 Beam Search (Size={cur_beam})")
                    continue
                except: print("格式错误，请用: beam 5"); continue
            
            if not text: continue

            # 执行分词与编码
            tokens = tokenize_chinese_hanlp(text)
            src_ids = [zh_vocab.word2idx['<SOS>']] + zh_vocab.encode(tokens) + [zh_vocab.word2idx['<EOS>']]
            src_tensor = torch.tensor([src_ids]).to(device)

            # 推理
            with torch.no_grad():
                if cur_method == 'beam':
                    decoded_ids, _ = model.beam_decode(src_tensor, beam_size=cur_beam, max_len=args.max_len)
                else:
                    decoded_ids, _ = model.greedy_decode(src_tensor, max_len=args.max_len)

            result = en_vocab.decode(decoded_ids[0].tolist(), skip_special=True)
            print(f"[英文] >>> {' '.join(result)}")

        except Exception as e:
            print(f"❌ 出错: {e}")

# ==================== 3. 主函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=['rnn', 'transformer'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--cache_dir', type=str, default='./cache/')
    parser.add_argument('--frequency', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--decode_method', type=str, default='beam', choices=['greedy', 'beam'])
    parser.add_argument('--beam_size', type=int, default=5)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_obj, zh_v, en_v = load_model(args, device)
    start_inference(model_obj, zh_v, en_v, device, args)