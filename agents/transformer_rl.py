import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)].to(x.device)


class TransformerRLAgent(nn.Module):
    def __init__(self, state_dim, n_actions, d_model=64, n_heads=4, n_layers=2, max_len=20):
        super().__init__()
        self.embedding = nn.Embedding(state_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=128)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.q_head = nn.Linear(d_model, n_actions)

    def forward(self, state_sequence):
        """
        state_sequence: (batch_size, seq_len) of token IDs (state indices)
        Returns: Q-values for the last token in the sequence
        """
        # Embed + add positional encoding
        x = self.embedding(state_sequence)  # (batch, seq_len, d_model)
        x = self.pos_enc(x)

        # Transformer expects (seq_len, batch, d_model)
        x = x.permute(1, 0, 2)
        encoded = self.encoder(x)  # (seq_len, batch, d_model)
        last_state = encoded[-1]   # get encoding of last state in sequence

        q_values = self.q_head(last_state)  # (batch, n_actions)
        return q_values
