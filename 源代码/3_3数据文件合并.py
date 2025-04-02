import os
import pandas as pd

def merge_excel_files(folder_path, output_file):
    # 用于存储所有 DataFrame 的列表
    all_dfs = []
    # 遍历指定文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.xlsx'):
                file_path = os.path.join(root, file)
                try:
                    # 读取 Excel 文件到 DataFrame
                    df = pd.read_excel(file_path)
                    all_dfs.append(df)
                    print(f"成功读取文件: {file_path}")
                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}")
    if all_dfs:
        # 按行合并所有 DataFrame
        merged_df = pd.concat(all_dfs, ignore_index=True)
        try:
            # 将合并后的 DataFrame 保存到新的 Excel 文件中
            merged_df.to_excel(output_file, index=False)
            print(f"合并后的数据已保存到: {output_file}")
        except Exception as e:
            print(f"保存合并后的数据到 {output_file} 时出错: {e}")
    else:
        print("未找到有效的 Excel 文件。")


# 指定包含 Excel 文件的文件夹路径
folder_path = '..\Lib\Data\Sample_xlsx'
# 指定合并后文件的保存路径
output_file = '..\Lib\Data\Sample.xlsx'

# 调用函数进行合并
merge_excel_files(folder_path, output_file)