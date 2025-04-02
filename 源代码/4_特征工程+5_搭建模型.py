import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM as LSTM_Keras
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv


# 0. 将 GeoLife 数据集中的 Date 和 Time 列合并并转换为时间戳
def merge_and_convert_to_timestamp(data):
    """

    参数:
    data (pandas.DataFrame): 包含 Date 和 Time 列的 GeoLife 数据集

    返回:
    pandas.DataFrame: 包含合并后时间戳列的 GeoLife 数据集
    """
    # 检查数据框中是否存在 Date 和 Time 列
    if 'Date' not in data.columns or 'Time' not in data.columns:
        print("数据框中缺少 'Date' 或 'Time' 列，请检查数据。")
        return data

    # 合并 Date 和 Time 列
    data['datetime'] = data['Date'] + ' ' + data['Time']

    # 将合并后的列转换为 datetime 类型
    data['datetime'] = pd.to_datetime(data['datetime'])

    # 将 datetime 类型转换为时间戳（以秒为单位）
    data['timestamp'] = data['datetime'].astype('int64') // 10**9

    return data

# 1. 数据读取
def read_data(file_path):
    data = pd.read_excel(file_path)
    data = merge_and_convert_to_timestamp(data)
    return data

# 2. 特征工程
def feature_engineering(data):
    # 提取时间特征
    data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
    data['hour'] = data['datetime'].dt.hour
    data['day_of_week'] = data['datetime'].dt.dayofweek

    # 计算位置偏移量
    data['delta_Latitude'] = data['Latitude'].diff()
    data['delta_Longitude'] = data['Longitude'].diff()

    # 处理缺失值
    data = data.dropna()

    # 定义特征和目标
    features = data[['Latitude', 'Longitude', 'hour', 'day_of_week', 'delta_Latitude', 'delta_Longitude']]
    target = data[['Latitude', 'Longitude']].shift(-1)  # 预测下一个位置

    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 移除最后一行的 NaN（由于 shift）
    features = features[:-1]
    target = target.dropna()

    return features, target

# 3. 滑动窗口内的平均移动速度
def calculate_avg_speed(data, window_size=3):
    """
    计算滑动窗口内的平均移动速度
    :param data: 包含经纬度和时间戳的 DataFrame
    :param window_size: 滑动窗口大小
    :return: 包含平均移动速度列的 DataFrame
    """
    distances = []
    times = []
    for i in range(len(data) - 1):
        coord1 = (data.iloc[i]['Latitude'], data.iloc[i]['Longitude'])
        coord2 = (data.iloc[i + 1]['Latitude'], data.iloc[i + 1]['Longitude'])
        distance = geodesic(coord1, coord2).meters
        time_diff = data.iloc[i + 1]['timestamp'] - data.iloc[i]['timestamp']
        # 避免除零错误
        if time_diff == 0:
            time_diff = 1e-6
        distances.append(distance)
        times.append(time_diff)
    speeds = np.array(distances) / np.array(times)
    avg_speeds = pd.Series(speeds).rolling(window=window_size).mean()
    data['avg_speed'] = [np.nan] + avg_speeds.tolist()  # 第一个元素设为 NaN
    return data

# 4. 历史访问位置频次（使用 sklearn.feature_extraction.DictVectorizer 编码）
def encode_location_frequency(data):
    """
    对历史访问位置频次进行编码
    :param data: 包含经纬度的 DataFrame
    :return: 编码后的 DataFrame
    """
    location_dict = []
    for _, row in data.iterrows():
        location = (row['Latitude'], row['Longitude'])
        # 将经纬度元组转换为字符串
        location_str = str(location)
        location_dict.append({'location': location_str})
    vectorizer = DictVectorizer()
    location_freq = vectorizer.fit_transform(location_dict)

    # 将稀疏矩阵与原始数据合并
    data_sparse = pd.DataFrame.sparse.from_spmatrix(location_freq, columns=vectorizer.get_feature_names_out())
    data.reset_index(drop=True, inplace=True)
    data_sparse.reset_index(drop=True, inplace=True)
    data = pd.concat([data, data_sparse], axis=1)

    return data

# 5. XGBoost 模型
def train_xgboost(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"XGBoost 模型得分: {score}")
    print(f"XGBoost 模型 MSE: {mse}")
    print(f"XGBoost 模型 RMSE: {rmse}")
    return model

# 6. LSTM 模型
def train_lstm(features, target):
    X = features.reshape((features.shape[0], features.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(LSTM_Keras(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    # 修改输出层神经元数量为目标数据的列数
    model.add(Dense(target.shape[1]))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    score = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"LSTM 模型损失: {score}")
    print(f"LSTM 模型 MSE: {mse}")
    print(f"LSTM 模型 RMSE: {rmse}")
    return model
# 7. Seq2Seq 模型
def train_seq2seq(features, target, timesteps=10):
    X = np.array([features[i:i+timesteps] for i in range(len(features) - timesteps)])
    y = np.array([target[i:i+timesteps] for i in range(len(target) - timesteps)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    encoder_inputs = Input(shape=(timesteps, features.shape[1]))
    encoder = LSTM_Keras(50, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(timesteps, 2))  # 假设目标是经纬度
    decoder_lstm = LSTM_Keras(50, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(2)
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='mse')

    model.fit([X_train, y_train], y_train, epochs=10, batch_size=32, verbose=1)
    score = model.evaluate([X_test, y_test], y_test)
    y_pred = model.predict([X_test, y_test])
    mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    print(f"Seq2Seq 模型损失: {score}")
    print(f"Seq2Seq 模型 MSE: {mse}")
    print(f"Seq2Seq 模型 RMSE: {rmse}")
    return model

# 8. GCN 模型
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

def train_gcn(features, target):
    # 检查输入数据类型并转换为张量
    if isinstance(features, pd.DataFrame):
        features = features.values
    if isinstance(target, pd.DataFrame):
        target = target.values

    # 将输入数据转换为 PyTorch 张量
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(target, dtype=torch.float)

    # 简单示例：按时间顺序连接相邻节点
    num_nodes = x.shape[0]
    edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes - 1)], dtype=torch.long).t().contiguous()

    # 动态设置模型输出维度以匹配目标
    out_channels = y.shape[1]  # 目标维度（例如，经纬度为2）
    model = GCN(in_channels=x.shape[1], hidden_channels=16, out_channels=out_channels)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 训练循环
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    y_pred = model(x, edge_index).detach().numpy()
    y_true = y.numpy()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"GCN 模型最终损失: {loss.item()}")
    print(f"GCN 模型 MSE: {mse}")
    print(f"GCN 模型 RMSE: {rmse}")
    return model

# 主函数
def main():class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

def train_gcn(features, target):
    # 检查输入数据类型并转换为张量
    if isinstance(features, pd.DataFrame):
        features = features.values
    if isinstance(target, pd.DataFrame):
        target = target.values

    # 将输入数据转换为 PyTorch 张量
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(target, dtype=torch.float)

    # 简单示例：按时间顺序连接相邻节点
    num_nodes = x.shape[0]
    edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes - 1)], dtype=torch.long).t().contiguous()

    # 动态设置模型输出维度以匹配目标
    out_channels = y.shape[1]  # 目标维度（例如，经纬度为2）
    model = GCN(in_channels=x.shape[1], hidden_channels=16, out_channels=out_channels)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 训练循环
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    y_pred = model(x, edge_index).detach().numpy()
    y_true = y.numpy()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"GCN 模型最终损失: {loss.item()}")
    print(f"GCN 模型 MSE: {mse}")
    print(f"GCN 模型 RMSE: {rmse}")
    return model
    file_path = '..\Lib\Data\Sample.xlsx'
    data = read_data(file_path)
    features, target = feature_engineering(data)

    # 构建高阶特征
    df1 = calculate_avg_speed(data)
    print(df1)
    df2 = encode_location_frequency(data)
    print(df2)

    # 训练 XGBoost 模型
    xgboost_model = train_xgboost(features, target)

    # 训练 LSTM 模型
    lstm_model = train_lstm(features, target)

    # 训练 Seq2Seq 模型
    seq2seq_model = train_seq2seq(features, target)

    # 训练 GCN 模型
    gcn_model = train_gcn(features, target)

if __name__ == "__main__":
    main()