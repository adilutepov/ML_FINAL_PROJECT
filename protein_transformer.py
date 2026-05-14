import torch
import torch.nn as nn


AA_VOCAB = {
    "A": 1, "C": 2, "D": 3, "E": 4,
    "F": 5, "G": 6, "H": 7, "I": 8,
    "K": 9, "L": 10, "M": 11, "N": 12,
    "P": 13, "Q": 14, "R": 15, "S": 16,
    "T": 17, "V": 18, "W": 19, "Y": 20
}

def encode_protein(seq, max_len=512):

    seq = seq[:max_len]

    encoded = [AA_VOCAB.get(aa, 0) for aa in seq]

    while len(encoded) < max_len:
        encoded.append(0)

    return torch.tensor(encoded, dtype=torch.long)

class ProteinTransformer(nn.Module):

    def __init__(self,
                 vocab_size=25,
                 emb_dim=128,
                 nhead=4,
                 num_layers=2):

        super().__init__()

        # amino acid embedding
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # pooling → vector
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):

        # x: (batch, seq_len)
        x = self.embedding(x)

        # transformer
        x = self.transformer(x)

        # pooling
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)

        return self.fc(x)