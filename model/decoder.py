from torch import nn
from model.attention import MultiHeadAttention
from model.feedforward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff_sublayer = FeedForward(d_model, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, norm_x, norm_x, src_mask)
        x = x + self.dropout(attn_output)
        norm_x = self.norm2(x)
        ff_output = self.ff_sublayer(norm_x)
        x = x + self.dropout(ff_output)
        return x