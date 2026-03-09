import pandas as pd
from statsmodels.regression.linear_model import OLS
import numpy as np


# 加载股票数据和市场因子数据
stock_file_path = "C:\\Users\\A\\Desktop\\160105\\selected_merged_file0105.csv"
market_file_path = "C:\\Users\\A\\Desktop\\160105\\Nikkei 225\\Nikkei225.csv"

# 读取数据
stock_data = pd.read_csv(stock_file_path)
market_data = pd.read_csv(market_file_path)

# 重命名市场因子数据的列名以便合并
market_data_renamed = market_data.copy()
market_data_renamed.columns = ['time', 'Market_Return']

# 获取股票数据中第一个时间列的名称（用于合并）
time_col_name = [col for col in stock_data.columns if 'col1' in col][0]

# 数据合并前的准备
stock_data_for_merge = stock_data.copy()
stock_data_for_merge['time'] = stock_data_for_merge[time_col_name]

# 合并数据
stock_data = pd.merge(stock_data_for_merge,
                      market_data_renamed,
                      on='time',
                      how='left')

# 对Market_Return减去0.05%
stock_data['Market_Return'] = stock_data['Market_Return'] - 0.0005

# 删除临时创建的time列
stock_data = stock_data.drop('time', axis=1)

# 提取股票数目
n_stocks = stock_data.shape[1] // 2  # 每只股票有两列数据

# 初始化预测结果存储的字典
prediction_columns = {}

# 对每只股票进行预测
for i in range(n_stocks):
    # 获取当前股票的时间和滞后收益率
    time_col = stock_data.iloc[:, 2 * i]
    returns_col = stock_data.iloc[:, 2 * i + 1]
    market_returns = stock_data['Market_Return']

    # 删除任何包含缺失值的行
    valid_mask = returns_col.notna() & market_returns.notna()
    returns_col = returns_col[valid_mask]
    market_returns = market_returns[valid_mask]

    # 初始化预测结果
    forecast = [np.nan] * len(stock_data)

    # 确保有足够的数据进行预测
    if len(returns_col) > 3:
        # 创建滞后变量
        lag1 = returns_col.shift(1)
        lag2 = returns_col.shift(2)
        lag3 = returns_col.shift(3)

        # 删除包含NaN的行
        df = pd.DataFrame({
            'y': returns_col,
            'lag1': lag1,
            'lag2': lag2,
            'lag3': lag3,
            'market': market_returns
        }).dropna()

        if len(df) > 0:
            # 准备自变量（X）和因变量（y）
            X = df[['lag1', 'lag2', 'lag3', 'market']]
            y = df['y']

            # 添加常数项
            X = pd.concat([pd.Series(1, index=X.index), X], axis=1)
            X.columns = ['const', 'lag1', 'lag2', 'lag3', 'market']

            # 拟合模型
            model = OLS(y, X)
            results = model.fit()

            # 获取系数
            alpha = results.params['const']
            beta1 = results.params['lag1']
            beta2 = results.params['lag2']
            beta3 = results.params['lag3']
            gamma1 = results.params['market']

            # 进行预测
            for t in range(3, len(returns_col)):
                if (not np.isnan(returns_col.iloc[t - 1]) and
                        not np.isnan(returns_col.iloc[t - 2]) and
                        not np.isnan(returns_col.iloc[t - 3]) and
                        not np.isnan(market_returns.iloc[t])):
                    pred = (alpha +
                            beta1 * returns_col.iloc[t - 1] +
                            beta2 * returns_col.iloc[t - 2] +
                            beta3 * returns_col.iloc[t - 3] +
                            gamma1 * market_returns.iloc[t])

                    # 将预测值放入对应的位置
                    forecast[valid_mask.index[t]] = pred

    # 保存预测结果到字典
    new_column_name = f"Stock_{i + 1}_Prediction"
    prediction_columns[new_column_name] = forecast

# 将预测结果插入到原始股票数据中
output_data = stock_data_for_merge.copy()  # 使用原始股票数据创建输出数据框
for i in range(n_stocks):
    new_column_name = f"Stock_{i + 1}_Prediction"
    output_data.insert(2 * i + 2 + i, new_column_name, prediction_columns[new_column_name])

# 删除预测列中083000之前的预测值，只保留083000及之后的预测数据
for i in range(n_stocks):
    prediction_col_name = f"Stock_{i + 1}_Prediction"
    output_data.loc[output_data[time_col_name] < 83000, prediction_col_name] = np.nan

# 删除临时创建的time列（如果存在）
if 'time' in output_data.columns:
    output_data = output_data.drop('time', axis=1)

# 保存结果为CSV文件
output_file_path = "C:\\Users\\A\\Desktop\\160105\\AR3和Market混合模型预测.csv"
output_data.to_csv(output_file_path, index=False)

# 输出数据的前几行进行检查
print("预测结果的前几行：")
print(output_data.head())
