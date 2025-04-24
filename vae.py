
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取时间序列数据
data_path = "data/data.xlsx"
df = pd.read_excel(data_path, sheet_name='sheet2', usecols=[2, 3, 4,5,6,7])

# 数据预处理，将时间序列数据归一化到 [0, 1] 范围内
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

# VAE的编码和解码过程
class SimpleVAE:
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_weights = np.random.rand(input_dim, latent_dim)
        self.decoder_weights = np.random.rand(latent_dim, input_dim)

    def encode(self, x):
        return np.dot(x, self.encoder_weights)

    def decode(self, z):
        return np.dot(z, self.decoder_weights)

    def fit(self, data, epochs=1000):
        for epoch in range(epochs):
            # 编码和解码
            z = self.encode(data)
            reconstructed = self.decode(z)

            # 计算损失
            loss = np.mean((data - reconstructed) ** 2)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 定义参数
input_dim = scaled_data.shape[1]
latent_dim = 5
epochs = 500

# 创建并训练VAE模型
vae = SimpleVAE(input_dim, latent_dim)
vae.fit(scaled_data, epochs)

# 使用VAE模型进行异常检测
reconstructed_data = vae.decode(vae.encode(scaled_data))
mse = np.mean((scaled_data - reconstructed_data) ** 2, axis=1)
threshold = np.percentile(mse, 95)  # 根据置信度设置阈值
anomalies = np.where(mse > threshold)[0]
# 反归一化
original_data = scaler.inverse_transform(reconstructed_data)
# df.values
# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(original_data[:, 0], label='重构数据', color='r')
plt.plot(df.values[:, 0], label='原始数据', color='b')
plt.scatter(anomalies, df.values[anomalies, 0], marker='o', color='g', label='异常点')
plt.legend()
plt.title('使用简单VAE进行时间序列异常检测')
plt.show()

# 标记异常样本
y_pred = np.zeros_like(mse)
y_pred[mse > threshold] = 1  # 异常样本标签为1

# 假设 mse 和 y_pred 是一维数组，original_data 是二维数组
results = pd.DataFrame({
    'MSE': mse,                    # 均方误差列
    'Anomaly': y_pred,              # 异常预测列
    'chonggoushuju': original_data[:, 0]  # 提取 original_data 的第一列
})

# 保存结果到 Excel 文件
results.to_excel('vae_anomaly_detection_results.xlsx', index=False)
