import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以便结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        # 均值
        self.fc_mu = nn.Linear(128, latent_dim)
        # log方差(取log可以保证方差始终为正数, 符合数学定义)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

    # 编码器
    def encoder(self, x):
        x = torch.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    # 重参数化技巧
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码器
    def decoder(self, z):
        z = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(z))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# 数据加载和预处理
def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# 训练循环
def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader, start=1):

        data = data.to(device).view(data.size(0), -1)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = criterion(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}")
    return train_loss / len(train_loader.dataset)


# 评估模型
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device).view(data.size(0), -1)
            recon_batch, mu, logvar = model(data)
            test_loss += criterion(recon_batch, data, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")
    return test_loss


# VAE损失函数
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# 可视化生成图像
def visualize_generated_images(model, num_images=10, device="cpu"):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, model.fc_mu.out_features).to(device)
        generated_images = model.decoder(z)
        generated_images = generated_images.view(num_images, 1, 28, 28).cpu().numpy()

    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
    for i in range(num_images):
        axes[i].imshow(generated_images[i][0], cmap='gray')
        axes[i].axis('off')
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    input_dim = 784
    latent_dim = 20
    batch_size = 128
    num_epochs = 50
    learning_rate = 1e-4

    model = VAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader, test_loader = load_data(batch_size)

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, vae_loss, device, epoch)
        test_loss = evaluate(model, test_loader, vae_loss, device)

        # 保存模型检查点
        if epoch % 10 == 0:  # 每隔5个epoch保存一次
            checkpoint_path = f"./VAE/result/vae_model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")

    # 训练完成后保存最终模型
    final_model_path = "vae_model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    visualize_generated_images(model, num_images=10, device=device)
