import pandas as pd
import os

# 定义数据处理函数
def process_file(file_path, output_dir):
    # 加载CSV文件
    df = pd.read_csv(file_path)

    # 选择需要的列：第6列（股票编号）、第8列（交易时间）、第18列（best ask）、第21列（best bid）
    df_selected = df.iloc[:, [5, 7, 17, 20]].copy()  # 使用copy()方法创建副本

    # 确保第8列（交易时间）是字符串格式
    df_selected.iloc[:, 1] = df_selected.iloc[:, 1].astype(str)

    # 标准化时间格式，使所有时间为六位数
    def standardize_time_format(time_str):
        return time_str.zfill(6)  # 在前面补零，确保时间为六位数

    df_selected.iloc[:, 1] = df_selected.iloc[:, 1].apply(standardize_time_format)

    # 打印标准化后的时间预览
    print("\n标准化后的时间列预览：")
    print(df_selected.iloc[:, 1].head())

    # 计算第18列和第21列的平均值，生成一个新的列 'price'
    df_selected['price'] = (df_selected.iloc[:, 2] + df_selected.iloc[:, 3]) / 2

    # 打印计算后的price列
    print("\n计算后的price列预览：")
    print(df_selected.iloc[:, [2, 3, 4]].head())

    # 定义函数返回秒数与整分钟的差值
    def seconds_difference_to_whole_minute(time_str):
        try:
            seconds = int(time_str[-2:])  # 提取最后两位作为秒数
            return min(seconds, 60 - seconds)  # 计算与整分钟（0或60秒）的差值
        except ValueError:
            return float('inf')  # 返回一个大数值以确保错误数据不会被选择

    # 为每个时间段找到最接近整分钟的记录
    df_selected['hour_minute'] = df_selected.iloc[:, 1].str[:4]  # 提取前四个字符作为小时和分钟部分

    # 使用groupby和apply方法来找到每组内最接近整分钟的记录
    df_selected['seconds_diff'] = df_selected.iloc[:, 1].apply(seconds_difference_to_whole_minute)
    df_filtered = df_selected.loc[df_selected.groupby('hour_minute')['seconds_diff'].idxmin()]

    # 删除临时分组列
    df_filtered = df_filtered.drop(columns=['hour_minute', 'seconds_diff'])

    # 按照第8列（交易时间）进行升序排序
    df_filtered = df_filtered.sort_values(by=df_filtered.columns[1])

    # 生成输出文件路径
    base_name = os.path.basename(file_path)  # 获取文件名
    output_file_name = f'processed_{base_name}'  # 设置输出文件名
    output_file_path = os.path.join(output_dir, output_file_name)  # 合成输出文件路径

    # 保存处理后的数据到新的CSV文件
    df_filtered.to_csv(output_file_path, index=False)

    print(f"处理后的数据已保存到 {output_file_path}")

# 指定输入和输出目录
input_dir = r"E:\\stock and lag return\\0105测试数据"
output_dir = r"E:\\stock and lag return\\0105"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取输入目录中所有CSV文件的路径
input_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.csv')]

# 对每个文件执行处理
for file_path in input_files:
    process_file(file_path, output_dir)
