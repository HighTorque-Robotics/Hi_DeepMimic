import shutil
import os

def copy_py_files(source_folder, destination_folder):
    # 若目标文件夹不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.py'):
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_folder, file)
                try:
                    shutil.copy2(source_file_path, destination_file_path)
                    # print(f"成功复制 {source_file_path} 到 {destination_file_path}")
                except Exception as e:
                    print(f"复制 {source_file_path} 时出错: {e}")
