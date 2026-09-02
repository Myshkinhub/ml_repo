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
def generate_stripe_data(num_samples=4000, img_size=16):
    """
    生成条纹图像数据集。
    返回：图像 (batch, 1, img_size, img_size) 和标签 (batch,)
    标签：0 = 横向条纹，1 = 纵向条纹
    """
    images = []
    labels = []
    for _ in range(num_samples):
        # 随机选择条纹方向
        label = np.random.randint(0, 2)  # 0或1
        # 创建全零图像
        img = np.zeros((img_size, img_size), dtype=np.float32)
        if label == 0:  # 横向条纹：每隔一行填255
            for i in range(0, img_size, 2):
                img[i, :] = 255.0
        else:  # 纵向条纹：每隔一列填255
            for j in range(0, img_size, 2):
                img[:, j] = 255.0
        # 归一化到 [0,1]
        img = img / 255.0
        images.append(img)
        labels.append(label)

    # 转为张量
    X = torch.FloatTensor(images).unsqueeze(1)  # (样本, 1, 高, 宽)
    y = torch.LongTensor(labels)

    # 划分训练集和测试集
    split = int(0.8 * num_samples)
    train_data = TensorDataset(X[:split], y[:split])
    test_data = TensorDataset(X[split:], y[split:])
    return train_data, test_data

# 2.定义一个简单的CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个卷积层：输入通道1，输出通道8，卷积核3x3，填充1保持尺寸
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # 池化层：2x2最大池化，尺寸减半
        self.pool = nn.MaxPool2d(2, 2)
        # 第二个卷积层：输入8，输出16，卷积核3x3
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # 全连接层：经过两次池化，图像尺寸从16->8->4，所以特征图为 16 * 4 * 4 = 256
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 2)  # 二分类输出

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 16->8
        x = self.pool(torch.relu(self.conv2(x)))  # 8->4
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # 不经过softmax，因为CrossEntropyLoss自带
        return x

# 3.训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=30):
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
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Test Acc: {acc * 100:.2f}%')
    return train_losses, test_accs

# 4.可视化
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

def show_sample_images(train_data, num=8):
    """显示几张训练样本，帮助理解数据"""
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for i in range(num):
        img, label = train_data[i]
        ax = axes[i // 4, i % 4]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title('Horizontal' if label == 0 else 'Vertical')
        ax.axis('off')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # 短暂暂停，让窗口正常显示

#  5.主程序
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数
    batch_size = 64
    learning_rate = 0.001
    epochs = 30

    # 生成数据
    train_data, test_data = generate_stripe_data()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # 显示一些样本（直观了解数据）
    show_sample_images(train_data)

    # 创建模型
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练
    print("开始训练...")
    losses, accs = train_model(model, train_loader, test_loader, criterion, optimizer, epochs)

    # 绘制训练曲线
    plot_results(losses, accs)

    # 手动测试：预测一张新生成的条纹图
    model.eval()
    # 生成一个横向条纹
    h_img = np.zeros((16, 16), dtype=np.float32)
    for i in range(0, 16, 2):
        h_img[i, :] = 1.0
    # 生成一个纵向条纹
    v_img = np.zeros((16, 16), dtype=np.float32)
    for j in range(0, 16, 2):
        v_img[:, j] = 1.0

    for name, img in [('Horizontal', h_img), ('Vertical', v_img)]:
        x = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).item()
        print(f"{name} image -> predicted as: {'Horizontal' if pred == 0 else 'Vertical'}")