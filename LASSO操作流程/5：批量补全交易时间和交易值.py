#在原数据基础上补全交易时间与交易值，用就近补全的方法（处理整个文件夹里面的文件）
import pandas as pd
import numpy as np
import os

# 定义输入和输出文件夹路径
input_folder = "E:\\stock and lag return\\0105"
output_folder = "E:\\stock and lag return\\3"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有 CSV 文件
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        # 生成文件的完整路径
        file_path = os.path.join(input_folder, filename)

        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 删除第三列与第四列数据
        df.drop(columns=df.columns[2:4], inplace=True)

        # 清理第二列（交易时间）中的非数字字符，转换为字符串类型并补全为六位数
        df.iloc[:, 1] = df.iloc[:, 1].replace({r'\D': '0'}, regex=True).fillna('000000')
        df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.zfill(6)

        # 删除不在交易时间范围内的记录
        df = df[(df.iloc[:, 1] >= '080000') & (df.iloc[:, 1] <= '113000') |
                (df.iloc[:, 1] >= '123000') & (df.iloc[:, 1] <= '150000')]

        # 补全前四行数据，使用前向填充
        df.iloc[:4] = df.iloc[:4].ffill()

        # 删除第一行数据
        df = df.iloc[1:].reset_index(drop=True)

        # 补全第二行数据的交易值与股票代码
        df.iloc[1] = df.iloc[1].ffill()

        # 创建完整的时间序列
        morning_times = pd.date_range("08:00:00", "11:30:00", freq="min").strftime('%H%M%S').tolist()
        afternoon_times = pd.date_range("12:30:00", "15:00:00", freq="min").strftime('%H%M%S').tolist()
        complete_times = morning_times + afternoon_times

        # 找出缺失的时间点
        existing_times = df.iloc[:, 1].tolist()
        missing_times = sorted(set(complete_times) - set(existing_times))

        # 在原数据上补全缺失的时间点，插入空行
        for time in missing_times:
            new_row = pd.DataFrame({df.columns[0]: [np.nan], df.columns[1]: [time], df.columns[2]: [np.nan]})
            df = pd.concat([df, new_row], ignore_index=True)

        # 重新排序 DataFrame 以确保时间顺序
        df = df.sort_values(by=df.columns[1])

        # 对第一列（股票代码）和第三列（交易值）进行前向填充
        df.iloc[:, 0] = df.iloc[:, 0].ffill()
        df.iloc[:, 2] = df.iloc[:, 2].ffill()

        # 只保留每分钟内的一个交易时间，筛选出最接近分钟开始时间的记录
        df['分钟'] = df.iloc[:, 1].str[:4]  # 提取时间的前四位（小时和分钟）
        df = df.groupby('分钟', as_index=False).apply(lambda x: x.loc[x.iloc[:, 1].idxmin()])

        # 删除辅助列“分钟”
        df.drop(columns=['分钟'], inplace=True)

        # 生成输出文件的完整路径
        output_file_path = os.path.join(output_folder, filename)

        # 保存处理后的数据到新的 CSV 文件
        df.to_csv(output_file_path, index=False)

        print(f"文件 {filename} 已处理并保存到 {output_file_path}")
