#.plt数据转.xlsx
import pandas as pd
import os

def plt_to_xls(plt_file_path, xls_file_path):
    # 跳过前 6 行元数据，读取实际的轨迹数据
    df = pd.read_csv(plt_file_path, skiprows=6, header=None)
    # 为数据框添加列名
    df.columns = ['Latitude', 'Longitude', '0', 'Altitude', 'NumDays', 'Date', 'Time']
    # 将数据保存为 .xlsx 文件
    df.to_excel(xls_file_path, index=False)

def convert_all_plt_to_xls(plt_folder, xls_folder):
    # 检查保存 .xlsx 文件的文件夹是否存在，不存在则创建
    if not os.path.exists(xls_folder):
        os.makedirs(xls_folder)
    # 遍历 .plt 文件夹中的所有文件
    for root, dirs, files in os.walk(plt_folder):
        for file in files:
            if file.endswith('.plt'):
                plt_file_path = os.path.join(root, file)
                # 构建对应的 .xlsx 文件路径
                xlsx_file_name = os.path.splitext(file)[0] + '.xlsx'
                xls_file_path = os.path.join(xls_folder, xlsx_file_name)
                try:
                    # 调用转换函数进行转换
                    plt_to_xls(plt_file_path, xls_file_path)
                    print(f"成功将 {plt_file_path} 转换为 {xls_file_path}")
                except Exception as e:
                    print(f"转换 {plt_file_path} 时出错: {e}")

# 指定 .plt 文件所在的文件夹路径
plt_folder = r'..\Lib\Data\Sample'
# 指定保存 .xlsx 文件的文件夹路径
xls_folder = r'..\Lib\Data\Sample_xlsx'

# 调用函数进行批量转换
convert_all_plt_to_xls(plt_folder, xls_folder)