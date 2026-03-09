import os
import pandas as pd

# 定义输入和输出文件路径
input_file = "C:\\Users\\A\\Desktop\\160105\\Nikkei 225\\processed_Nikkei 225 （1471）.csv"  # 替换为您要处理的文件路径
output_file = "C:\\Users\\A\\Desktop\\160105\\Nikkei 225\\Nikkei225.csv"  # 输出文件路径

# 加载CSV文件
data = pd.read_csv(input_file)

# 检查并填充前几行的缺失值
for i in range(min(50, len(data))):  # 检查前50行
    if data.iloc[i].isnull().any():
        # 找到下一个不为空的行进行填充
        for j in range(i + 1, len(data)):
            if not data.iloc[j].isnull().any():
                data.iloc[i] = data.iloc[i].fillna(data.iloc[j])
                break

# 将第三列重命名为 'price'
data.rename(columns={data.columns[2]: 'price'}, inplace=True)

# 尝试从第一列中提取股票代码，如果前几行都为NaN，则寻找有效值
stock_code = None
for i in range(len(data)):
    if not pd.isna(data.iloc[i, 0]):
        stock_code = str(int(data.iloc[i, 0]))
        break

if stock_code is None:
    raise ValueError(f"文件 {input_file} 中没有有效的股票代码")

# 将第二列重命名为股票代码（即交易价格列）
data.rename(columns={data.columns[1]: stock_code}, inplace=True)

# 删除第一列（原股票代码列）
data.drop(columns=[data.columns[0]], inplace=True)

# 计算滞后收益率并添加新列
data['lagged_return'] = data['price'].pct_change()

# 创建只包含交易时间和滞后收益率的DataFrame
data_final = data[[stock_code, 'lagged_return']].copy()

# 将交易时间列的名称更改为股票代码
data_final.rename(columns={stock_code: 'Time'}, inplace=True)
data_final.rename(columns={'Time': stock_code}, inplace=True)

# 保存更新后的数据为CSV文件
data_final.to_csv(output_file, index=False)

print(f"处理并保存了文件: {output_file}")
