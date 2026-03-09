import pandas as pd
import os

# 文件路径
input_file = r"E:\\stock and lag return\\201601output\\HTICST110.20160105.1"
output_dir = r"E:\\stock and lag return\\201601output\\0105股票"

# 读取完整数据
data = pd.read_csv(input_file, sep=',', header=None, dtype=str)

# 确定股票代码所在的列索引（第六列，索引为5）
stock_code_column = 5

# 获取所有唯一的股票代码
stock_codes = data[stock_code_column].unique()

# 输出股票数量
print(f"文件中包含 {len(stock_codes)} 只股票。")

# 创建输出目录（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历每个股票代码，分割数据并保存为单独的CSV文件
for stock_code in stock_codes:
    # 提取对应股票代码的数据
    stock_data = data[data[stock_code_column] == stock_code]

    # 定义输出文件路径
    output_file = os.path.join(output_dir, f"stock_{stock_code}.csv")

    # 保存为CSV文件，保持原始数据格式
    stock_data.to_csv(output_file, index=False, header=False)

    print(f"股票代码 {stock_code} 的数据已保存为: {output_file}")

print("所有股票数据已分割并保存为单独的CSV文件。")
