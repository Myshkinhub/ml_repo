import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use('TkAgg')
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
    生成二分类数据：序列是否单调递增。
    返回：序列 (batch, seq_len, 1) 和 标签 (batch,)
    """
    sequences = []
    labels = []
    for _ in range(num_samples):
        # 随机生成序列：从[-1,1]均匀分布采样
        seq = np.random.uniform(-1, 1, seq_len)
        # 判断是否严格递增（每个元素大于前一个）
        is_increasing = all(seq[i] < seq[i+1] for i in range(seq_len-1))
        sequences.append(seq)
        labels.append(1 if is_increasing else 0)
    # 转为张量
    X = torch.FloatTensor(sequences).unsqueeze(2)  # (样本, 步长, 特征数=1)
    y = torch.LongTensor(labels)
    # 划分训练集和测试集
    split = int(0.8 * num_samples)
    train_data = TensorDataset(X[:split], y[:split])
    test_data = TensorDataset(X[split:], y[split:])
    return train_data, test_data

# 2.定义RNN模型
class SimpleRNNClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, output_size=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐状态
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)          # out: (batch, seq_len, hidden)
        # 取最后一个时间步的输出
        out = out[:, -1, :]               # (batch, hidden)
        out = self.fc(out)                # (batch, 2)
        return out

# 3.训练和评估
def train(model, train_loader, test_loader, criterion, optimizer, epochs=50):
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

        # 测试集准确率
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
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Acc: {acc*100:.2f}%')
    return train_losses, test_accs

# 4.可视化
def plot_curves(losses, accs):
    print("accs =", accs[:10])
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

# 5.主程序
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数
    seq_len = 10
    batch_size = 64
    hidden_size = 16
    num_layers = 1
    learning_rate = 0.001
    epochs = 50

    # 生成数据
    train_data, test_data = generate_data(seq_len=seq_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # 创建模型
    model = SimpleRNNClassifier(input_size=1, hidden_size=hidden_size,
                                num_layers=num_layers, output_size=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练
    print("开始训练...")
    losses, accs = train(model, train_loader, test_loader, criterion, optimizer, epochs)

    # 绘制曲线
    plot_curves(losses, accs)

    # 最后测试一个样本，验证效果
    model.eval()
    test_seq = np.random.uniform(-1, 1, seq_len)
    # 构造一个递增序列用于测试
    inc_seq = np.sort(np.random.uniform(-1, 1, seq_len))
    for name, seq in [('随机序列', test_seq), ('递增序列', inc_seq)]:
        x = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(2).to(device)
        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).item()
        print(f"{name} -> 预测类别: {'递增' if pred==1 else '非递增'}")