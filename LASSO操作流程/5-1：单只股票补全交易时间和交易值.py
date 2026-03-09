import pandas as pd
import numpy as np
import os

# 定义输入和输出文件路径
input_file = "C:\\Users\\A\\Desktop\\160105\\Nikkei 225\\processed_Nikkei 225 （1471）.csv"  # 替换为你要处理的单个文件路径
output_file_path = "C:\\Users\\A\\Desktop\\160105\\Nikkei 225\\Nikkei 225.csv"  # 指定输出文件的完整路径

# 确保输出文件的目录存在
output_dir = os.path.dirname(output_file_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# 读取 CSV 文件
df = pd.read_csv(input_file)

# 打印原始列信息（调试用）
print(f"原始列数: {len(df.columns)}, 列名: {df.columns.tolist()}")

# 删除第三列与第四列数据（索引 2 和 3）
if len(df.columns) >= 4:
    df.drop(columns=df.columns[2:4], inplace=True)
    print("已删除第三列和第四列。")
else:
    print("警告：DataFrame 列数少于4，无法删除第三列和第四列。")

# 确认删除后的列数（调试用）
print(f"删除列后的列数: {len(df.columns)}, 列名: {df.columns.tolist()}")

# 清理第二列（交易时间）中的非数字字符，转换为字符串类型并补全为六位数
# 假设第二列名为 '交易时间'，如果没有列名，请根据实际情况修改
交易时间列名 = df.columns[1]
df[交易时间列名] = df[交易时间列名].astype(str).str.replace(r'\D', '0', regex=True).fillna('000000').str.zfill(6)

# 打印标准化后的时间预览
print("\n标准化后的时间列预览：")
print(df[交易时间列名].head())

# 删除不在交易时间范围内的记录
df = df[
    ((df[交易时间列名] >= '080000') & (df[交易时间列名] <= '113000')) |
    ((df[交易时间列名] >= '123000') & (df[交易时间列名] <= '150000'))
]

print(f"过滤后的行数: {len(df)}")

# 补全前四行数据，使用前向填充
if len(df) >= 4:
    df.iloc[:4] = df.iloc[:4].ffill()
    print("已完成前四行的前向填充。")
else:
    df = df.ffill()
    print("DataFrame 行数少于4，已对所有行进行前向填充。")

# 删除第一行数据
if len(df) >= 1:
    df = df.iloc[1:].reset_index(drop=True)
    print("已删除第一行数据。")
else:
    print("警告：DataFrame 没有足够的行来删除第一行。")

# 补全第二行数据的交易值与股票代码
if len(df) >= 2:
    df.iloc[1] = df.iloc[1].ffill()
    print("已补全第二行的数据。")
else:
    df = df.ffill()
    print("DataFrame 行数少于2，已对所有行进行前向填充。")

# 创建完整的时间序列
morning_times = pd.date_range("08:00:00", "11:30:00", freq="min").strftime('%H%M%S').tolist()
afternoon_times = pd.date_range("12:30:00", "15:00:00", freq="min").strftime('%H%M%S').tolist()
complete_times = morning_times + afternoon_times

# 找出缺失的时间点
existing_times = df[交易时间列名].tolist()
missing_times = sorted(set(complete_times) - set(existing_times))
print(f"缺失的时间点数量: {len(missing_times)}")

# 在原数据上补全缺失的时间点，插入空行
for time in missing_times:
    # 动态生成新行，根据当前DataFrame的列数
    new_row_data = {col: np.nan for col in df.columns}
    new_row_data[交易时间列名] = time
    new_row = pd.DataFrame([new_row_data])
    df = pd.concat([df, new_row], ignore_index=True)

print("已插入所有缺失的时间点。")

# 重新排序 DataFrame 以确保时间顺序
df = df.sort_values(by=交易时间列名).reset_index(drop=True)

# 对第一列（假设为 '股票代码'）和第三列（假设为 '交易值'）进行前向填充
股票代码列名 = df.columns[0]
交易值列名 = df.columns[-1]  # 最后一列假设为 '交易值'

df[股票代码列名] = df[股票代码列名].ffill()
df[交易值列名] = df[交易值列名].ffill()

# 只保留每分钟内的一个交易时间，筛选出最接近分钟开始时间的记录
# 提取时间的前四位（小时和分钟）
df['分钟'] = df[交易时间列名].str[:4]

# 定义函数返回秒数与整分钟的差值
def seconds_difference_to_whole_minute(time_str):
    try:
        seconds = int(time_str[-2:])
        return min(seconds, 60 - seconds)
    except ValueError:
        return float('inf')  # 对无法转换的值返回一个大值

df['seconds_diff'] = df[交易时间列名].apply(seconds_difference_to_whole_minute)

# 使用 groupby 找到每组中 seconds_diff 最小的记录
df_filtered = df.loc[df.groupby('分钟')['seconds_diff'].idxmin()].reset_index(drop=True)

# 删除辅助列“分钟”和“seconds_diff”
df_filtered.drop(columns=['分钟', 'seconds_diff'], inplace=True)

# 保存处理后的数据到新的 CSV 文件
df_filtered.to_csv(output_file_path, index=False)

print(f"文件 {input_file} 已处理并保存到 {output_file_path}")
