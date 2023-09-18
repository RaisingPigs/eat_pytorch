import torch.nn as nn


class ClassifyNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # 设置padding_idx参数后将在训练过程中将填充的token始终赋值为0向量
        # 输入shape [64, 200]
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=3, padding_idx=0)
        # 经过embedding后 shape [64, 200, 3]
        # 需要再经过transpose(1,2) shape变为 [64, 3, 200]

        # 因为是1维卷积, 所以kernel_size只能是单个值
        self.conv = nn.Sequential(
            # 经过Conv1d后 shape [64, 16, 196]
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5),
            # 经过MaxPool1d后 shape [64, 16, 98]
            nn.MaxPool1d(kernel_size=2),
            nn.GELU(),
            # 经过Conv1d后 shape [64, 128, 94]
            nn.Conv1d(in_channels=16, out_channels=128, kernel_size=5),
            # 经过MaxPool1d后 shape [64, 128, 47]
            nn.MaxPool1d(kernel_size=2),
            nn.GELU()
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            # 经过Linear后 shape [64, 1]
            nn.Linear(128 * 47, 1)
        )

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        y = self.dense(x)
        return y
