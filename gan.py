import matplotlib
matplotlib.use('TkAgg')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus']=False  # 负号显示

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 将图像归一化到 [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# 定义生成器网络
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_dim=784):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, img_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), 1, 28, 28)


# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出概率值
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)


# 初始化模型
latent_dim = 100
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
lr = 0.0002
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


# 训练函数
def train_gan(epochs):
    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(train_loader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # 创建标签
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ---------------------
            #  训练判别器
            # ---------------------
            d_optimizer.zero_grad()

            # 计算判别器对真实图像的损失
            real_pred = discriminator(real_imgs)
            d_real_loss = criterion(real_pred, real_labels)

            # 生成假图像
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)

            # 计算判别器对假图像的损失
            fake_pred = discriminator(fake_imgs.detach())
            d_fake_loss = criterion(fake_pred, fake_labels)

            # 总判别器损失
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # ---------------------
            #  训练生成器
            # ---------------------
            g_optimizer.zero_grad()

            # 生成假图像
            fake_imgs = generator(z)
            # 计算判别器对假图像的预测
            fake_pred = discriminator(fake_imgs)
            # 生成器希望判别器将假图像判断为真
            g_loss = criterion(fake_pred, real_labels)
            g_loss.backward()
            g_optimizer.step()

            # 打印训练进度
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(train_loader)} "
                      f"Discriminator Loss: {d_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")

        # 每个epoch结束后，生成一些样本图像
        if (epoch + 1) % 10 == 0:
            generate_samples(generator, epoch + 1, latent_dim, device)


# 生成样本图像
def generate_samples(generator, epoch, latent_dim, device, n_samples=16):
    os.makedirs("gan_samples", exist_ok=True)   # <-- 新增这一行
    generator.eval()
    z = torch.randn(n_samples, latent_dim).to(device)
    with torch.no_grad():
        samples = generator(z).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i][0].numpy(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    # 更稳妥的文件名构造
    if isinstance(epoch, int):
        filename = f"gan_samples/gan_samples_epoch_{epoch}.png"
    else:
        filename = f"gan_samples/gan_samples_{epoch}.png"
    plt.savefig(filename)
    plt.close()
    generator.train()


# 训练模型
train_gan(epochs=50)

# 生成最终样本
generate_samples(generator, "final", latent_dim, device)


