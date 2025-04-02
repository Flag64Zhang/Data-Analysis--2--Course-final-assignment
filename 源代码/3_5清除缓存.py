import shutil
import os
def delete_folder(base_folder):

    # 定义要删除的文件夹列表
    folders_to_delete = ['Sample', 'Sample_xlsx']

    # 遍历要删除的文件夹列表
    for folder in folders_to_delete:
        folder_path = os.path.join(base_folder, folder)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                 # 使用 shutil.rmtree 删除文件夹及其内容
              shutil.rmtree(folder_path)
              print(f"成功删除文件夹: {folder_path}")
            except Exception as e:
                print(f"删除文件夹 {folder_path} 时出错: {e}")
        else:
              print(f"文件夹 {folder_path} 不存在。")

def delete_sample(base_folder):

    # 定义要检测和删除的文件名
    file_name = "Sample.xlsx"
    # 构建完整的文件路径
    file_path = os.path.join(base_folder, file_name)

    # 检查文件是否存在
    if os.path.exists(file_path) and os.path.isfile(file_path):
        try:
            # 若文件存在，执行删除操作
            os.remove(file_path)
            print(f"已成功删除文件: {file_path}")
        except Exception as e:
            # 捕获删除过程中可能出现的异常并打印错误信息
            print(f"删除文件 {file_path} 时出现错误: {e}")
    else:
        # 若文件不存在，给出相应提示
        print(f"文件 {file_path} 不存在。")


# 定义基础文件夹路径
base_folder = r"..\Lib\Data"
delete_folder(base_folder)
delete_sample(base_folder)