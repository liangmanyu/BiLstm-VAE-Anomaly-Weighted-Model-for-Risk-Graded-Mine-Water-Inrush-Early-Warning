import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载与归一化
data_path = "data/data1.xlsx"
df = pd.read_excel(data_path, sheet_name='sheet2', usecols=[2, 3, 4, 5, 6, 7])
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)
data_tensor = torch.tensor(scaled_data, dtype=torch.float32)

# 构建VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(8, latent_dim)
        self.logvar_layer = nn.Linear(8, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

# 损失函数
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# 超参数设置
input_dim = scaled_data.shape[1]
latent_dim = 3
epochs = 100
batch_size = 32
learning_rate = 0.001

# 模型训练
model = VAE(input_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
dataloader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        x_batch = batch[0]
        reconstructed, mu, logvar = model(x_batch)
        loss = loss_function(reconstructed, x_batch, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# 计算重构误差和检测异常
model.eval()
with torch.no_grad():
    reconstructed, _, _ = model(data_tensor)
    mse = torch.mean((data_tensor - reconstructed) ** 2, dim=1).numpy()
    threshold = np.percentile(mse, 95)
    anomalies = np.where(mse > threshold)[0]
    reconstructed_np = reconstructed.numpy()
    original_data = scaler.inverse_transform(reconstructed_np)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(df.values[:, 0], label='原始数据', color='b')
plt.plot(original_data[:, 0], label='重构数据', color='r')
plt.scatter(anomalies, df.values[anomalies, 0], color='g', label='异常点', marker='o')
plt.legend()
plt.title('基于VAE的时间序列异常检测')
plt.show()

# 保存结果
y_pred = np.zeros_like(mse)
y_pred[mse > threshold] = 1

results = pd.DataFrame({
    'MSE': mse,
    'Anomaly': y_pred,
    'Reconstructed': original_data[:, 0]
})
results.to_excel('vae_anomaly_detection_results.xlsx', index=False)
