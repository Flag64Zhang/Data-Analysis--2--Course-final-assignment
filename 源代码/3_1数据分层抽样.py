import os
import random
import shutil
# 基础文件夹路径
base_folder = r'..\Lib\Data'
# 新建的 Sample 文件夹路径
sample_folder = os.path.join(base_folder, "Sample")

# 创建 Sample 文件夹
if not os.path.exists(sample_folder):
    os.makedirs(sample_folder)

# 循环从 1 到 180
for i in range(1, 182):
    # 将数字格式化为三位字符串，不足三位时前面补 0
    subfolder_name = f"{i:03d}\Trajectory"
    # 构建完整的子文件夹路径
    folder_path = os.path.join(base_folder, subfolder_name)

    # 检查文件夹是否存在
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"正在处理文件夹: {folder_path}")
        try:
            # 获取该文件夹下所有 .plt 文件
            plt_files = [f for f in os.listdir(folder_path) if f.endswith('.plt')]
            if plt_files:
                # 随机选择一个 .plt 文件
                random_plt_file = random.choice(plt_files)
                source_file_path = os.path.join(folder_path, random_plt_file)
                destination_file_path = os.path.join(sample_folder, random_plt_file)
                # 复制选中的 .plt 文件到 Sample 文件夹
                shutil.copy2(source_file_path, destination_file_path)
                print(f"  已将 {random_plt_file} 复制到 {sample_folder}")
            else:
                print("  该文件夹下没有 .plt 文件。")
        except PermissionError:
            print(f"  没有权限访问文件夹 {folder_path} 中的内容。")
        except Exception as e:
            print(f"  处理文件夹 {folder_path} 时出现错误: {e}")
    else:
        print(f"文件夹 {folder_path} 不存在。")