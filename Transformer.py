import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use('TkAgg')

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


# 1.生成数据
def generate_data(num_samples=5000, seq_len=10):
    """
    任务：判断序列中偶数（0,2,4,6,8）是否占多数。
    标签：1 = 偶数占多数，0 = 奇数占多数。
    """
    sequences = []
    labels = []
    for _ in range(num_samples):
        # 生成 0~9 的随机整数序列
        seq = np.random.randint(0, 10, size=seq_len)
        # 统计偶数个数
        even_count = np.sum(seq % 2 == 0)
        label = 1 if even_count > seq_len / 2 else 0
        sequences.append(seq)
        labels.append(label)

    X = torch.LongTensor(np.array(sequences))  # 整数序列，适合 Embedding
    y = torch.LongTensor(np.array(labels))

    # 划分训练/测试集
    split = int(0.8 * num_samples)
    train_data = TensorDataset(X[:split], y[:split])
    test_data = TensorDataset(X[split:], y[split:])
    return train_data, test_data


# 2.位置编码（Transformer 必备）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return x


# 3.定义 Transformer 模型
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=10, d_model=32, nhead=4, num_layers=2, max_len=20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # 将数字0~9映射为向量
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=False,  # 输入形状为 (seq, batch, feature)
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model) -> Transformer 要求的格式
        x = self.pos_encoder(x)  # 添加位置信息
        x = self.transformer_encoder(x)  # (seq_len, batch, d_model)

        # 取所有时间步的平均值（聚合全局信息）
        x = x.mean(dim=0)  # (batch, d_model)
        x = self.fc(x)  # (batch, 2) 二分类
        return x


# 4.训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50):
    train_losses, test_accs = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                _, predicted = torch.max(pred, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        acc = correct / total
        test_accs.append(acc)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Test Acc: {acc * 100:.2f}%')
    return train_losses, test_accs


# 5.可视化
def plot_results(losses, accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2.plot(accs, marker='o', linestyle='-', color='red', linewidth=2, markersize=4)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


# 6.主程序
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数
    seq_len = 10
    batch_size = 64
    epochs = 30  # 这个任务很简单，30轮足够

    # 生成数据
    train_data, test_data = generate_data(seq_len=seq_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # 创建模型
    model = SimpleTransformer(vocab_size=10, d_model=32, nhead=4, num_layers=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    print("开始训练 Transformer...")
    losses, accs = train_model(model, train_loader, test_loader, criterion, optimizer, epochs)

    # 绘制训练曲线
    plot_results(losses, accs)

    # 7.手动测试（验证效果）
    model.eval()
    # 构造两个测试序列
    even_major = [2, 4, 6, 8, 0, 1, 3, 5, 7,
                  9]  # 偶数占多数（5个偶数 vs 5个奇数？等一下，这个有5个偶数(2,4,6,8,0)和5个奇数(1,3,5,7,9)，不是多数。改成6个偶数）
    even_major = [2, 4, 6, 8, 0, 1, 3, 5, 7, 9]  # 让我重写：偶数 >5个
    even_major = [2, 4, 6, 8, 0, 2, 4, 5, 7, 9]  # 偶数6个(2,4,6,8,0,2,4) 等等 0,2,4,6,8,2,4 共7个 -> 标签1
    even_major = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]  # 5个偶数，5个奇数（平局，但代码中 > 一半，即 >5 才为1，这个为0）
    # 重新明确构造：
    test_even = [0, 2, 4, 6, 8, 0, 2, 1, 3, 5]  # 偶数: 0,2,4,6,8,0,2 = 7个 -> 标签 1
    test_odd = [1, 3, 5, 7, 9, 1, 3, 0, 2, 4]  # 奇数: 1,3,5,7,9,1,3 = 7个 -> 标签 0

    for name, seq in [("偶数占多数 (应预测1)", test_even),
                      ("奇数占多数 (应预测0)", test_odd)]:
        x = torch.LongTensor(seq).unsqueeze(0).to(device)  # (1, seq_len)
        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).item()
        print(f"{name} -> 预测结果: {pred} ({'偶数多数' if pred == 1 else '奇数多数'})")