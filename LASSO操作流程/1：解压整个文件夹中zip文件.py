#所有操作中的第一步，先将整个文件夹中的zip文件进行解压，并进行保存
import os
import zipfile


def unzip_files_in_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                print(f'Unzipped: {zip_path}')


def main():
    input_dir = "E:\\百度网盘下载\\201602"  # 修改为你的输入目录路径
    output_dir = "E:\\stock and lag return\\201602output"  # 修改为你的输出目录路径
    unzip_files_in_directory(input_dir, output_dir)


if __name__ == '__main__':
    main()

