import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import FunctionTransformer

# 1. 数据读取
def read_xlsx_file(file_path):
    # 读取 .xlsx 文件
    data = pd.read_excel(file_path)
    return data

# 2. 数据清洗
def clean_data(data):
    # 处理缺失值
    data = data.dropna()
    return data

# 3. 异常点检测
def detect_outliers(data):
    # 提取经度和纬度作为特征
    X = data[['Latitude', 'Longitude']].values  # 转换为 numpy 数组
    # 使用孤立森林算法检测异常点
    clf = IsolationForest(contamination=0.05)  # 假设 5% 的数据是异常点
    clf.fit(X)
    # 预测时也使用 numpy 数组
    data['is_outlier'] = clf.predict(X)
    return data

# 4. 停留点识别
def identify_stay_points(data, distance_threshold=0.001, time_threshold=10):
    stay_points = []
    current_stay_point = []
    for i in range(len(data) - 1):
        # 计算两点之间的距离
        distance = np.sqrt((data.iloc[i]['Latitude'] - data.iloc[i + 1]['Latitude'])**2 +
                           (data.iloc[i]['Longitude'] - data.iloc[i + 1]['Longitude'])**2)
        if distance < distance_threshold:
            current_stay_point.append(data.iloc[i])
        else:
            if len(current_stay_point) >= time_threshold:
                stay_points.extend(current_stay_point)
            current_stay_point = []
    if len(current_stay_point) >= time_threshold:
        stay_points.extend(current_stay_point)
    stay_points_df = pd.DataFrame(stay_points)
    data['is_stay_point'] = data.index.isin(stay_points_df.index)
    return data

# 5. 数据可视化
def visualize_data(data):
    plt.figure(figsize=(12, 8))
    # 绘制正常点
    normal_points = data[data['is_outlier'] == 1]
    plt.scatter(normal_points['Longitude'], normal_points['Latitude'], c='b', label='Normal Points')
    # 绘制异常点
    outliers = data[data['is_outlier'] == -1]
    plt.scatter(outliers['Longitude'], outliers['Latitude'], c='r', label='Outliers')
    # 绘制停留点
    stay_points = data[data['is_stay_point']]
    plt.scatter(stay_points['Longitude'], stay_points['Latitude'], c='g', label='Stay Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Trajectory with Outliers and Stay Points')
    plt.legend()
    plt.show()

# 主函数
def main():
    file_path = '..\Lib\Data\Sample.xlsx'  # .xlsx 文件路径
    data = read_xlsx_file(file_path)
    data = clean_data(data)
    data = detect_outliers(data)
    data = identify_stay_points(data)
    visualize_data(data)

if __name__ == "__main__":
    main()