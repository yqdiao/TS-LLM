#整个文件夹里面补全前几行缺少数据并标准化数据后求滞后收益率，并删除股票代码列，将交易列列名改为股票代码，
# 删除price列，生成的数据只剩两列，一列交易时间，一列滞后收益率；
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 定义输入和输出文件夹路径
input_folder = "E:\\lag return\\stock and lag return\\Retail_Stocks"
output_folder = "C:\\Users\\A\\Desktop\\160105\\0105Retail data"

# 获取输入文件夹中所有CSV文件的列表
files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# 遍历每个文件进行操作
for file in files:
    # 构建完整的文件路径
    file_path = os.path.join(input_folder, file)

    # 加载CSV文件
    data = pd.read_csv(file_path)

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

    # 尝试从第一列中提取股票代码
    stock_code = None
    for i in range(len(data)):
        if not pd.isna(data.iloc[i, 0]):
            stock_code = str(int(data.iloc[i, 0]))
            break

    if stock_code is None:
        raise ValueError(f"文件 {file} 中没有有效的股票代码")

    # 将第二列重命名为股票代码（即交易价格列）
    data.rename(columns={data.columns[1]: stock_code}, inplace=True)

    # 删除第一列（原股票代码列）
    data.drop(columns=[data.columns[0]], inplace=True)

    # 对价格数据进行标准化处理
    scaler = StandardScaler()
    # 确保价格数据为数值类型
    data['price'] = pd.to_numeric(data['price'], errors='coerce')
    # 移除可能的无穷大值
    data['price'] = data['price'].replace([np.inf, -np.inf], np.nan)
    # 保存非NaN的索引
    valid_indices = ~data['price'].isna()

    if valid_indices.any():
        # 只对非NaN值进行标准化
        price_values = data.loc[valid_indices, 'price'].values.reshape(-1, 1)
        standardized_prices = scaler.fit_transform(price_values)
        data.loc[valid_indices, 'price'] = standardized_prices.flatten()

    # 计算标准化后价格的滞后收益率
    data['lagged_return'] = data['price'].pct_change()

    # 创建只包含交易时间和滞后收益率的DataFrame
    data_final = data[[stock_code, 'lagged_return']].copy()

    # 将交易时间列的名称更改为股票代码
    data_final.rename(columns={stock_code: 'Time'}, inplace=True)
    data_final.rename(columns={'Time': stock_code}, inplace=True)

    # 构建输出文件路径
    output_file_path = os.path.join(output_folder, f"{stock_code}.csv")

    # 保存更新后的数据为CSV文件
    data_final.to_csv(output_file_path, index=False)

    print(f"处理并保存了文件: {output_file_path}")