import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal

# ==================== 1. 位置编码模块 ====================

class AbsolutePositionalEncoding(nn.Module):
    """
    标准正弦余弦位置编码 (Absolute Positional Encoding)
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码 (Relative Positional Encoding)
    在注意力计算中加入相对位置偏置
    """
    def __init__(self, d_model: int, num_heads: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # 相对位置偏置表: 从 -max_len+1 到 max_len-1
        self.relative_bias = nn.Embedding(2 * max_len - 1, num_heads)
        
    def get_relative_positions(self, seq_len: int, device):
        # 生成相对位置矩阵
        range_vec = torch.arange(seq_len, device=device)
        relative_pos = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)  # (seq_len, seq_len)
        relative_pos = relative_pos + self.max_len - 1  # 偏移到正数范围
        relative_pos = relative_pos.clamp(0, 2 * self.max_len - 2)
        return relative_pos

    def forward(self, x, attn_scores=None):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        
        if attn_scores is not None:
            # 为注意力分数添加相对位置偏置
            relative_pos = self.get_relative_positions(seq_len, x.device)
            bias = self.relative_bias(relative_pos)  # (seq_len, seq_len, num_heads)
            bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, seq_len, seq_len)
            return attn_scores + bias
        
        return self.dropout(x)


# ==================== 2. 归一化模块 ====================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    比 LayerNorm 更简单高效，去掉了均值中心化
    """
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


def get_norm_layer(norm_type: str, d_model: int):
    """获取归一化层"""
    if norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    elif norm_type == "rmsnorm":
        return RMSNorm(d_model)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


# ==================== 3. 多头注意力模块 ====================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    支持相对位置编码
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float = 0.1,
        use_relative_pos: bool = False,
        max_len: int = 512
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_relative_pos = use_relative_pos
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        if use_relative_pos:
            self.relative_pos = RelativePositionalEncoding(d_model, num_heads, max_len)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 添加相对位置偏置
        if self.use_relative_pos:
            scores = self.relative_pos(query, scores)
        
        # 应用 mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights


# ==================== 4. 前馈网络模块 ====================

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ==================== 5. Transformer 编码器层 ====================

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        use_relative_pos: bool = False,
        max_len: int = 512
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_relative_pos, max_len)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = get_norm_layer(norm_type, d_model)
        self.norm2 = get_norm_layer(norm_type, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-Attention with residual connection
        attn_output, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        
        # Feed-Forward with residual connection
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        
        return src


# ==================== 6. Transformer 解码器层 ====================

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        use_relative_pos: bool = False,
        max_len: int = 512
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_relative_pos, max_len)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, False, max_len)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = get_norm_layer(norm_type, d_model)
        self.norm2 = get_norm_layer(norm_type, d_model)
        self.norm3 = get_norm_layer(norm_type, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, encoder_output, trg_mask=None, src_mask=None):
        # Masked Self-Attention
        attn_output, _ = self.self_attn(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + self.dropout(attn_output))
        
        # Cross-Attention
        attn_output, attn_weights = self.cross_attn(trg, encoder_output, encoder_output, src_mask)
        trg = self.norm2(trg + self.dropout(attn_output))
        
        # Feed-Forward
        ff_output = self.feed_forward(trg)
        trg = self.norm3(trg + self.dropout(ff_output))
        
        return trg, attn_weights


# ==================== 7. Transformer 编码器 ====================

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        pos_encoding: str = "absolute",
        norm_type: str = "layernorm",
        pad_idx: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.scale = math.sqrt(d_model)
        
        use_relative = (pos_encoding == "relative")
        
        if pos_encoding == "absolute":
            self.pos_encoding = AbsolutePositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoding = nn.Dropout(dropout)  # 相对位置在 attention 中处理
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, norm_type, use_relative, max_len)
            for _ in range(num_layers)
        ])
        
        self.norm = get_norm_layer(norm_type, d_model)

    def forward(self, src, src_mask=None):
        # src: (batch, src_len)
        x = self.embedding(src) * self.scale
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return self.norm(x)


# ==================== 8. Transformer 解码器 ====================

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        pos_encoding: str = "absolute",
        norm_type: str = "layernorm",
        pad_idx: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.scale = math.sqrt(d_model)
        
        use_relative = (pos_encoding == "relative")
        
        if pos_encoding == "absolute":
            self.pos_encoding = AbsolutePositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoding = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, norm_type, use_relative, max_len)
            for _ in range(num_layers)
        ])
        
        self.norm = get_norm_layer(norm_type, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, trg, encoder_output, trg_mask=None, src_mask=None):
        # trg: (batch, trg_len)
        x = self.embedding(trg) * self.scale
        x = self.pos_encoding(x)
        
        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(x, encoder_output, trg_mask, src_mask)
        
        x = self.norm(x)
        output = self.fc_out(x)
        
        return output, attn_weights


# ==================== 9. Transformer Seq2Seq ====================

class Transformer(nn.Module):
    def __init__(
        self,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        device,
        pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def make_src_mask(self, src):
        # src: (batch, src_len)
        # 返回: (batch, 1, 1, src_len) 用于广播
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # trg: (batch, trg_len)
        batch_size, trg_len = trg.shape
        
        # Padding mask
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Causal mask (下三角矩阵)
        trg_causal_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_causal_mask = trg_causal_mask.unsqueeze(0).unsqueeze(1)
        
        # 组合 mask
        trg_mask = trg_pad_mask & trg_causal_mask
        return trg_mask

    def forward(self, src, trg, teacher_forcing_ratio=None):
        """
        Transformer 训练时使用完整的 target 序列（并行计算）
        teacher_forcing_ratio 参数保留以保持接口一致，但不使用
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        encoder_output = self.encoder(src, src_mask)
        output, attn_weights = self.decoder(trg, encoder_output, trg_mask, src_mask)
        
        return output, attn_weights

    def greedy_decode(self, src, max_len=50):
        """贪婪解码"""
        batch_size = src.shape[0]
        src_mask = self.make_src_mask(src)
        encoder_output = self.encoder(src, src_mask)
        
        # 初始化解码序列
        decoded_ids = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=self.device)
        
        for _ in range(max_len - 1):
            trg_mask = self.make_trg_mask(decoded_ids)
            output, _ = self.decoder(decoded_ids, encoder_output, trg_mask, src_mask)
            
            # 取最后一个时间步的预测
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            decoded_ids = torch.cat([decoded_ids, next_token], dim=1)
            
            # 检查是否全部生成 EOS
            if (next_token == self.eos_idx).all():
                break
        
        return decoded_ids, None

    def beam_decode(self, src, beam_size=3, max_len=50):
        """
        Transformer 的 Beam Search 解码
        """
        batch_size = src.shape[0]
        vocab_size = self.decoder.fc_out.out_features
        
        # 1. 编码
        src_mask = self.make_src_mask(src)
        encoder_outputs = self.encoder(src, src_mask)
        
        # 2. 初始化
        beams_ids = torch.full((batch_size, beam_size, 1), self.sos_idx, 
                                dtype=torch.long, device=self.device)
        beams_scores = torch.zeros((batch_size, beam_size), device=self.device)
        beams_scores[:, 1:] = -1e9  # 初始只有第一个 beam 有效
        
        finished = torch.zeros((batch_size, beam_size), dtype=torch.bool, device=self.device)
        
        for t in range(1, max_len):
            # 当前所有 beam 的序列: (batch * beam, seq_len)
            current_seqs = beams_ids.view(batch_size * beam_size, -1)
            
            # 重复 encoder_outputs 给每个 beam
            enc_rep = encoder_outputs.repeat_interleave(beam_size, dim=0)
            src_mask_rep = src_mask.repeat_interleave(beam_size, dim=0)
            
            # Transformer decoder forward
            trg_mask = self.make_trg_mask(current_seqs)
            output, _ = self.decoder(current_seqs, enc_rep, trg_mask, src_mask_rep)  # ← 解包元组
            
            # 只取最后一个位置的预测: (batch * beam, vocab)
            last_output = output[:, -1, :]
            log_probs = F.log_softmax(last_output, dim=-1)
            log_probs = log_probs.view(batch_size, beam_size, vocab_size)
            
            # 处理已结束的 beam
            if finished.any():
                log_probs = log_probs.masked_fill(finished.unsqueeze(-1), float('-inf'))
                # 已结束的 beam 只能生成 pad
                pad_mask = finished.unsqueeze(-1).expand_as(log_probs).clone()
                pad_mask[:, :, self.pad_idx] = False
                log_probs = log_probs.masked_fill(pad_mask, float('-inf'))
                # pad token 的概率设为 0（不影响分数）
                log_probs[:, :, self.pad_idx] = torch.where(
                    finished, 
                    torch.zeros_like(log_probs[:, :, self.pad_idx]), 
                    log_probs[:, :, self.pad_idx]
                )
            
            # 计算新分数: (batch, beam, vocab)
            curr_scores = beams_scores.unsqueeze(2) + log_probs
            curr_scores = curr_scores.view(batch_size, -1)  # (batch, beam * vocab)
            
            # 选择 top-k
            top_scores, top_indices = curr_scores.topk(beam_size, dim=1)
            
            parent_beam_indices = top_indices // vocab_size  # (batch, beam)
            next_token_ids = top_indices % vocab_size        # (batch, beam)
            
            # 更新 beams_ids
            new_beams = torch.gather(
                beams_ids, 1, 
                parent_beam_indices.unsqueeze(-1).expand(-1, -1, beams_ids.size(2))
            )
            beams_ids = torch.cat([new_beams, next_token_ids.unsqueeze(2)], dim=2)
            
            # 更新分数
            beams_scores = top_scores
            
            # 更新 finished 状态
            finished = torch.gather(finished, 1, parent_beam_indices)
            finished = finished | (next_token_ids == self.eos_idx)
            
            # 如果所有 beam 都结束了，提前退出
            if finished.all():
                break
        
        # 选择最佳 beam
        best_beam_idx = beams_scores.argmax(dim=1)  # (batch,)
        results = torch.stack([beams_ids[b, best_beam_idx[b]] for b in range(batch_size)])
        
        # 填充到 max_len
        if results.size(1) < max_len:
            pad_tensor = torch.full(
                (batch_size, max_len - results.size(1)), 
                self.pad_idx, dtype=torch.long, device=self.device
            )
            results = torch.cat([results, pad_tensor], dim=1)
        
        return results, None


# ==================== 10. 模型构建函数 ====================

def build_model(
    src_vocab_size: int,
    trg_vocab_size: int,
    device,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ff: int = 2048,
    dropout: float = 0.1,
    max_len: int = 512,
    pos_encoding: str = "absolute",  # "absolute" or "relative"
    norm_type: str = "layernorm",    # "layernorm" or "rmsnorm"
    pad_idx: int = 0,
    sos_idx: int = 1,
    eos_idx: int = 2
):
    """
    构建 Transformer 模型
    
    Args:
        pos_encoding: 位置编码类型 - "absolute" (正弦余弦) 或 "relative" (相对位置)
        norm_type: 归一化类型 - "layernorm" 或 "rmsnorm"
    """
    encoder = TransformerEncoder(
        vocab_size=src_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        pos_encoding=pos_encoding,
        norm_type=norm_type,
        pad_idx=pad_idx
    )
    
    decoder = TransformerDecoder(
        vocab_size=trg_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        pos_encoding=pos_encoding,
        norm_type=norm_type,
        pad_idx=pad_idx
    )
    
    model = Transformer(encoder, decoder, device, pad_idx, sos_idx, eos_idx).to(device)
    
    # 参数初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # 打印模型配置
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"--- Transformer Model Config ---")
    print(f"d_model: {d_model} | num_heads: {num_heads} | num_layers: {num_layers}")
    print(f"d_ff: {d_ff} | dropout: {dropout} | max_len: {max_len}")
    print(f"Position Encoding: {pos_encoding} | Normalization: {norm_type}")
    print(f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")
    
    return model