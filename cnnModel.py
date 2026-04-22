import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_classes=2,
        num_filters=100,
        kernel_sizes=(3, 4, 5),
        dropout=0.5,
        pad_idx=0
    ):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)

        x = x.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2)
            conv_outputs.append(p)

        x = torch.cat(conv_outputs, dim=1)

        x = self.dropout(x)
        logits = self.fc(x)
        return logits