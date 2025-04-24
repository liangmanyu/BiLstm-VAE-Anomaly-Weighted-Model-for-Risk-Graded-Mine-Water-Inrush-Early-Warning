import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, explained_variance_score, mean_absolute_error, mean_squared_error, \
    r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, classification_report
from tensorflow.keras.optimizers import Adam, SGD, Adamax
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import GRU, LSTM, Bidirectional
from tensorflow.keras.layers import Concatenate, LeakyReLU
from tensorflow.keras.layers import Embedding, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

data = "data/data.xlsx"
df = pd.read_excel(data,sheet_name='Sheet1', usecols=[3,4,5,7], names=None)
X = df.iloc[:, 0:3]
Y = df.iloc[:, 3].values
print(X)
print(Y)
# # 对数据进行归一化
# 创建一个MinMaxScaler对象
scaler = MinMaxScaler()
Y = Y.reshape(-1, 1)
# 使用fit_transform函数将数据进行归一化
x = scaler.fit_transform(X)
y = scaler.fit_transform(Y)

# 计算训练集和测试集的大小
train_size = int(0.85 * len(x))
test_size = len(x) - train_size
# 划分训练集和测试集
X_train = x[:train_size,:]
X_test = x[train_size:,:]
y_train =y[:train_size]
y_test = y[train_size:]

# 扩展维度，3 维
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

start = time.time()
model = Sequential()
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(16, return_sequences=True, input_shape=(X_train.shape[1], 1))))
# model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(6, 1))))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(8, return_sequences=False)))
model.add(BatchNormalization())
model.add(Dense(10, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

# Adam 优化器
Adam = Adam(learning_rate=0.001, epsilon=1e-07)
# Adamax 优化器
Adamax = Adamax(learning_rate=0.01, beta_1=0.8, beta_2=0.888, epsilon=1e-07)
# 随机梯度下降 优化器
sgd = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

# Compile the model
model.compile(optimizer=Adam, loss='mse', metrics=['accuracy'])

# 保存模型结构
# tf.keras.utils.plot_model(model, "BRFSS_LSTM.png", show_shapes=True)

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,  # minimium amount of change to count as an improvement
    patience=20,  # how many epochs to wait before stopping
    restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, batch_size=72, epochs=50, validation_split=0.2)

# 预测
y_pred = model.predict(X_test)

#反归一化
y_test = scaler.inverse_transform(y_test)
y_pred = scaler.inverse_transform(y_pred)

# 将数据做成 列表 形式，这样就可以使用 MAE, MSE, RMSE, R2, MPAE
# y_pred 是一个嵌套的列表，不能直接做 MSE 等计算，所以需要把里面的数据 摘 出来
real_test_y = []
pred_test_y = []
for i in np.arange(y_pred.shape[0]):
    pred_test_y.append(int(y_pred[i]))

for i in np.arange(y_test.shape[0]):
    real_test_y.append(int(y_test[i]))

print("真实的 Y:", real_test_y)
print("预测的 Y:", pred_test_y)
print("LSTM 期望方差:", explained_variance_score(y_test, y_pred))   # 越接近 1 越好
print('LSTM 的 准确率 : {0:0.4f}'.format(accuracy_score(real_test_y, pred_test_y)))
print('LSTM 的 MAE = ', mean_absolute_error(real_test_y, pred_test_y)) # 越接近 0 越好
print('LSTM 的 MSE = ', mean_squared_error(real_test_y, pred_test_y))  # 越接近 0 越好
print('LSTM 的 RMSE = ', np.sqrt(mean_absolute_error(real_test_y, pred_test_y))) # 越接近 0 越好
print('LSTM 的 R2 = ', r2_score(real_test_y, pred_test_y)) # 越接近 1, 模型对数据拟合的越好
# MAPE 的值越小越好
print('LSTM 的 平均绝对百分比误差 MAPE = ', mean_absolute_percentage_error(real_test_y, pred_test_y))

# 将预测数据和真实数据 画出来
plt.plot(real_test_y, label='real value')
plt.plot(pred_test_y, label='predict value')
plt.xlabel('Days')
plt.ylabel('Predict numbers')
plt.legend()
plt.show()

# 记录结束时间，并显示
print("Runtime: ", time.time()-start, "秒")

#写入文件
y_test = pd.DataFrame(y_test)
y_pred = pd.DataFrame(y_pred)
writer = pd.ExcelWriter('LSTMyucejieguo.xlsx')  # 写入Excel文件
y_test.to_excel(writer, 'y_test', float_format='%.6f')  # ‘vif’是写入excel的sheet名
y_pred.to_excel(writer, 'y_pred', float_format='%.6f')  # ‘vif’是写入excel的sheet名
writer.save()
writer.close()

