# 代码中的数据集为CSV格式，包含多个特征。
# 首先将数据集分为训练集和测试集。
# 然后构建Isolation Forest模型并在训练集和测试集上进行预测并计算异常分数。
# 最后设置阈值并标记异常数据。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 加载数据集
data = "data/data2.xlsx"
df = pd.read_excel(data,sheet_name='sheet2', usecols=[1,2], names=None)
# 数据预处理
df = df.dropna()  # 去掉 null 数据
T = df.iloc[:, 0].values
Y = df.iloc[:, 1].values
# print(dataset.shape)
# print(dataset[:, 0])
# print(dataset[:, 1])
# # 将数据集分为训练集和测试集
train_data = Y.sample(frac=0.8, random_state=42)
test_data = Y.drop(train_data.index)

# 构建Isolation Forest模型
if_model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05)
if_model.fit(train_data)

# 在训练集上进行预测并计算异常分数
train_data_scores = -if_model.decision_function(train_data)

# 在测试集上进行预测并计算异常分数
test_data_scores = -if_model.decision_function(test_data)

# 设置阈值并标记异常数据
threshold = np.percentile(train_data_scores, 5)
test_data['Score'] = test_data_scores
test_data['Anomaly'] = np.where(test_data['Score'] < threshold, 1, 0)

# 输出异常数据
anomalies = test_data.loc[test_data['Anomaly'] == 1]
print(anomalies)

# 绘制散点图，用颜色区分正常和异常样本
# plt.scatter(dataset[:, 0], dataset[:, 1], c=anomalies, cmap='coolwarm')
plt.scatter(T, Y, c=anomalies, cmap='coolwarm')
plt.show()

