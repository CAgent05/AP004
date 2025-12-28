import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional, Literal

# ==================== 1. 注意力机制模块 ====================
class Attention(nn.Module):
    """
    实现三种对齐函数 (Alignment Functions):
    - dot: h_t^T * h_s
    - multiplicative: h_t^T * W * h_s
    - additive: v^T * tanh(W1*h_s + W2*h_t)
    """
    def __init__(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_type: Literal["dot", "multiplicative", "additive"] = "additive"
    ):
        super().__init__()
        self.attention_type = attention_type
        
        if attention_type == "multiplicative":
            self.W = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)
        elif attention_type == "additive":
            self.W1 = nn.Linear(encoder_hidden_dim, decoder_hidden_dim, bias=False)
            self.W2 = nn.Linear(decoder_hidden_dim, decoder_hidden_dim, bias=False)
            self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden: (batch, dec_hid) -> 取最后一层隐状态
        # encoder_outputs: (batch, src_len, enc_hid)
        
        if self.attention_type == "dot":
            # (batch, src_len, enc_hid) * (batch, enc_hid, 1) -> (batch, src_len)
            scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        
        elif self.attention_type == "multiplicative":
            # (batch, src_len, dec_hid)
            transformed_enc = self.W(encoder_outputs)
            scores = torch.bmm(transformed_enc, decoder_hidden.unsqueeze(2)).squeeze(2)
        
        elif self.attention_type == "additive":
            # (batch, src_len, dec_hid) + (batch, 1, dec_hid)
            res = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden).unsqueeze(1))
            scores = self.v(res).squeeze(2)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=1)
        # 上下文向量: (batch, enc_hid)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return attention_weights, context

# ==================== 2. 编码器 (Encoder) ====================
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=2, dropout=0.5, rnn_type="gru"):
        super().__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        rnn_class = nn.LSTM if rnn_type == "lstm" else nn.GRU
        # 要求：两层单向 RNN
        self.rnn = rnn_class(emb_dim, hidden_dim, n_layers, batch_first=True, 
                             dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

# ==================== 3. 解码器 (Decoder) ====================
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=2, dropout=0.5, 
                 rnn_type="gru", attention_type="additive"):
        super().__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attention = Attention(hidden_dim, hidden_dim, attention_type)
        
        rnn_class = nn.LSTM if rnn_type == "lstm" else nn.GRU
        # RNN输入：embedding + context
        self.rnn = rnn_class(emb_dim + hidden_dim, hidden_dim, n_layers, 
                             batch_first=True, dropout=dropout)
        
        # 输出层输入：hidden + context + embedding
        self.fc_out = nn.Linear(hidden_dim + hidden_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask=None):
        # input: (batch)
        embedded = self.dropout(self.embedding(input.unsqueeze(1)))
        
        # 获取用于计算注意力的隐状态 (取RNN最后一层的h)
        h_query = hidden[0][-1] if self.rnn_type == "lstm" else hidden[-1]
        
        # 计算注意力
        attn_weights, context = self.attention(h_query, encoder_outputs, mask)
        
        # 拼接作为RNN输入
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        # 最终预测
        combined = torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=1)
        prediction = self.fc_out(combined)
        
        return prediction, hidden, attn_weights

# ==================== 4. 整体模型 (Seq2Seq) ====================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx=0, sos_idx=1, eos_idx=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def create_mask(self, src):
        return src == self.pad_idx

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        实现对比策略：
        - teacher_forcing_ratio = 1.0 (Teacher Forcing)
        - teacher_forcing_ratio = 0.0 (Free Running)
        """
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        mask = self.create_mask(src)
        
        input = trg[:, 0] # 起始符 <SOS>
        for t in range(1, trg_len):
            prediction, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[:, t] = prediction
            
            # 策略切换
            is_teacher = random.random() < teacher_forcing_ratio
            input = trg[:, t] if is_teacher else prediction.argmax(1)
            
        return outputs, None

    # ==================== 解码策略 ====================
    def greedy_decode(self, src, max_len=50):
        batch_size = src.shape[0]
        encoder_outputs, hidden = self.encoder(src)
        mask = self.create_mask(src)
        
        decoded_ids = torch.full((batch_size, max_len), self.pad_idx, dtype=torch.long).to(self.device)
        input = torch.full((batch_size,), self.sos_idx, dtype=torch.long).to(self.device)
        decoded_ids[:, 0] = input
        
        for t in range(1, max_len):
            prediction, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            input = prediction.argmax(1)
            decoded_ids[:, t] = input
            if (input == self.eos_idx).all(): break
            
        return decoded_ids, None

    def beam_decode(self, src, beam_size=3, max_len=50):
        """
        支持 Batch 的 Beam Search 解码实现
        """
        batch_size = src.shape[0]
        vocab_size = self.decoder.fc_out.out_features
        
        # 1. 编码
        encoder_outputs, hidden = self.encoder(src)
        mask = self.create_mask(src)
        
        # 从 hidden 获取层数
        if isinstance(hidden, tuple):  # LSTM
            n_layers = hidden[0].shape[0]
            rnn_type = "lstm"
        else:  # GRU
            n_layers = hidden.shape[0]
            rnn_type = "gru"

        # 2. 准备束搜索所需的重复状态
        encoder_outputs = encoder_outputs.repeat_interleave(beam_size, dim=0)
        mask = mask.repeat_interleave(beam_size, dim=0)
        
        if rnn_type == "lstm":
            hidden = (hidden[0].repeat_interleave(beam_size, dim=1),
                    hidden[1].repeat_interleave(beam_size, dim=1))
        else:
            hidden = hidden.repeat_interleave(beam_size, dim=1)

        # 3. 初始化序列和分数
        beams_ids = torch.full((batch_size, beam_size, 1), self.sos_idx, dtype=torch.long).to(self.device)
        beams_scores = torch.zeros((batch_size, beam_size)).to(self.device)
        beams_scores[:, 1:] = -1e9 

        results = torch.full((batch_size, max_len), self.pad_idx, dtype=torch.long).to(self.device)

        for t in range(1, max_len):
            current_input = beams_ids[:, :, -1].view(-1)
            prediction, hidden, _ = self.decoder(current_input, hidden, encoder_outputs, mask)
            
            log_probs = F.log_softmax(prediction, dim=1)
            log_probs = log_probs.view(batch_size, beam_size, vocab_size)
            
            curr_scores = beams_scores.unsqueeze(2) + log_probs
            top_scores, top_indices = curr_scores.view(batch_size, -1).topk(beam_size, dim=1)
            
            parent_beam_indices = top_indices // vocab_size
            next_token_ids = top_indices % vocab_size
            
            new_beams = []
            for b in range(batch_size):
                new_beams.append(beams_ids[b, parent_beam_indices[b]])
            beams_ids = torch.stack(new_beams)
            beams_ids = torch.cat([beams_ids, next_token_ids.unsqueeze(2)], dim=2)
            
            beams_scores = top_scores
            
            # 更新 hidden 状态
            if rnn_type == "lstm":
                h, c = hidden
                h = h.view(n_layers, batch_size, beam_size, -1)
                c = c.view(n_layers, batch_size, beam_size, -1)
                new_h = torch.stack([h[:, b, parent_beam_indices[b]] for b in range(batch_size)], dim=1)
                new_c = torch.stack([c[:, b, parent_beam_indices[b]] for b in range(batch_size)], dim=1)
                hidden = (new_h.view(n_layers, -1, h.shape[-1]), 
                        new_c.view(n_layers, -1, c.shape[-1]))
            else:
                h = hidden.view(n_layers, batch_size, beam_size, -1)
                new_h = torch.stack([h[:, b, parent_beam_indices[b]] for b in range(batch_size)], dim=1)
                hidden = new_h.view(n_layers, -1, h.shape[-1])

        best_beams_idx = beams_scores.argmax(dim=1)
        for b in range(batch_size):
            results[b, :beams_ids.shape[2]] = beams_ids[b, best_beams_idx[b]]

        return results, None

# ==================== 5. 构建模型函数 ====================
def build_model(
    src_vocab_size, 
    trg_vocab_size, 
    device, 
    emb_dim=300, 
    hidden_dim=512, 
    rnn_type="gru", 
    attention_type="additive",
    pad_idx=0, sos_idx=1, eos_idx=2
):
    """
    通过此函数快速切换配置进行对比实验
    """
    encoder = Encoder(src_vocab_size, emb_dim, hidden_dim, n_layers=2, rnn_type=rnn_type)
    decoder = Decoder(trg_vocab_size, emb_dim, hidden_dim, n_layers=2, 
                      rnn_type=rnn_type, attention_type=attention_type)
    
    model = Seq2Seq(encoder, decoder, device, pad_idx, sos_idx, eos_idx).to(device)
    
    print(f"--- Model Config ---")
    print(f"RNN: {rnn_type.upper()} | Attention: {attention_type} | Layers: 2")
    return model