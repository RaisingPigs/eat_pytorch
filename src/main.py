# 经过Conv1d后 shape [64, 16, 196]
import torch
from torch import nn

c1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5)
# 经过MaxPool1d后 shape [64, 16, 195]
m1 = nn.MaxPool1d(kernel_size=2)
# 经过Conv1d后 shape [64, 128, 191]
c2 = nn.Conv1d(in_channels=16, out_channels=128, kernel_size=5)
# 经过MaxPool1d后 shape [64, 128, 47]
m2 = nn.MaxPool1d(kernel_size=2)

inputs = torch.randn([64, 3, 200])
x = c1(inputs)
x = m1(x)  # [64, 3, 196]
x = c2(x)  # [64, 3, 98]
x = m2(x)  # [64, 3, 94]
# [64, 3, 47]
print(x.shape)