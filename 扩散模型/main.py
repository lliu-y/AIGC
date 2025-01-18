import torch
from torch.utils.data import DataLoader, TensorDataset
from diffusion import Diffusion
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

def generate_synthetic_data(batch_size, state_dim, action_dim, num_samples=10000):
    """
    生成一个简单的合成数据集，其中每个状态和目标动作都是随机生成的。
    """
    states = torch.randn(num_samples, state_dim)
    actions = torch.randn(num_samples, action_dim)
    return TensorDataset(states, actions)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型超参数
    state_dim = 10        # 状态维度
    action_dim = 5        # 动作维度
    hidden_dim = 128      # 隐藏层维度
    T = 1000              # 扩散步骤数量
    time_dim = 16         # 时间维度（用于时间嵌入）
    loss_type = "l2"      # 使用 L2 损失
    batch_size = 64       # 批大小
    num_epochs = 1       # 训练周期数

    # 生成训练数据集
    dataset = generate_synthetic_data(batch_size, state_dim, action_dim)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = Diffusion(
        loss_type=loss_type,
        beta_schedule="linear",  # 线性 beta schedule
        clip_denoised=True,
        predict_epsion=True,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        T=T,
        device=device
    )

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        total_loss = 0

        for batch_idx, (state, target) in enumerate(train_loader):
            state, target = state.to(device), target.to(device)

            optimizer.zero_grad()  # 清除之前的梯度

            # 将目标作为 x_start 输入到模型中进行训练
            loss = model.training_step((state, target), batch_idx)

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 保存模型检查点
        if (epoch + 1) % 5 == 0:  # 每隔5个epoch保存一次
            checkpoint_path = f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")

    # 训练完成后保存最终模型
    final_model_path = "diffusion_model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
