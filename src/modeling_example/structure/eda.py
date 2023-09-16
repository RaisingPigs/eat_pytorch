import pandas as pd
import matplotlib.pyplot as plt
from utils import Config

train_df = pd.read_csv(Config.TRAIN_FILEPATH)
print(train_df.head())

# 统计死了和活着的人数
ax = train_df['Survived'].value_counts().plot(
    kind='bar',
    figsize=(12, 8),
    fontsize=15, 
    rot=0  # 设置X轴上的刻度标签旋转角度 0~360
)
# ax是matplotlib轴对象
ax.set_ylabel('Counts', fontsize=15)
ax.set_xlabel('Survived', fontsize=15)
plt.show()

# 探索年龄分布情况
ax = train_df.Age.plot(
    kind='hist',
    bins=20,  # 将age分为20个箱子, 把数据装进去
    figsize=(12,8),
    fontsize=15,
)

ax.set_ylabel('Frequency', fontsize=15)
ax.set_xlabel('Age', fontsize=15)
plt.show()

# 年龄和label(Survived)的相关性
ax = train_df.query('Survived == 0').Age.plot(
    kind='density',
    figsize=(12, 8),
    fontsize=15
)
train_df.query('Survived == 1').Age.plot(
    kind='density',
    figsize=(12, 8),
    fontsize=15
)
ax.legend(['Survived == 0', 'Survived == 1'], fontsize=12)
ax.set_ylabel('Density', fontsize=15)
ax.set_xlabel('Age', fontsize=15)
plt.show()

