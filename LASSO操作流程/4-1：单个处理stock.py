import pandas as pd
import os

# 定义数据处理函数
def process_file(input_file_path, output_file_path):
    # 加载CSV文件
    df = pd.read_csv(input_file_path)

    # 选择需要的列：第6列（股票编号）、第8列（交易时间）、第18列（best ask）、第21列（best bid）
    df_selected = df.iloc[:, [5, 7, 17, 20]].copy()

    # 将时间列先转换为字符串，再使用zfill进行补零
    # 使用临时变量来存储转换后的列，避免链式赋值问题
    time_col = df_selected.iloc[:, 1].astype(str).str.zfill(6)
    df_selected.iloc[:, 1] = time_col  # 单独赋值，避免 FutureWarning

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
            seconds = int(time_str[-2:])
            return min(seconds, 60 - seconds)
        except ValueError:
            return float('inf')

    # 为每个时间段找到最接近整分钟的记录
    df_selected['hour_minute'] = df_selected.iloc[:, 1].str[:4]
    df_selected['seconds_diff'] = df_selected.iloc[:, 1].apply(seconds_difference_to_whole_minute)

    df_filtered = df_selected.loc[df_selected.groupby('hour_minute')['seconds_diff'].idxmin()]

    # 删除临时分组列
    df_filtered = df_filtered.drop(columns=['hour_minute', 'seconds_diff'])

    # 按照第8列（交易时间）进行升序排序
    df_filtered = df_filtered.sort_values(by=df_filtered.columns[1])

    # 保存处理后的数据到新的CSV文件
    df_filtered.to_csv(output_file_path, index=False)

    print(f"处理后的数据已保存到 {output_file_path}")

# 指定输入和输出文件路径
input_file = r"C:\Users\A\Desktop\160105\Nikkei 225\Nikkei 225 （1471）.csv"
output_file = r"C:\Users\A\Desktop\160105\Nikkei 225\Nikkei1.csv"

# 确保输出文件的目录存在
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 处理文件
process_file(input_file, output_file)
