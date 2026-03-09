#计算每只股票LASSO预测值和基准模型预测值的均值和标准差，然后对这些预测值进行标准化处理
#以标准化后的LASSO预测值和基准模型预测值为自变量，实际收益率为因变量，构建线性回归模型。通过回归模型分析LASSO和基准模型对实际收益率的影响。
#通过比较回归模型在同时使用LASSO和基准模型（R²_both）与仅使用基准模型（R²_bmk）时的R²值，计算样本外拟合优度的增量
#计算投资组合的Sharpe比率，用于评估投资组合在风险调整后的收益表现，并将其添加到结果数据框中
#我们将LASSO的一分钟前收益预测标准化为零均值和单位方差，以便我们可以比较不同回归的斜率系数B（缺少系数的统计）

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 定义文件路径
lasso_file_path = "E:\\stock and lag return\\滚动预测0104.csv"
bmk_file_path = "E:\\stock and lag return\\基准模型预测0104.csv"
output_file_path = "E:\\stock and lag return\\regression_results.csv"

# 加载数据，只读取201列到301列
lasso_data = pd.read_csv(lasso_file_path, usecols=range(201, 301))
bmk_data = pd.read_csv(bmk_file_path, usecols=range(201, 301))

# 提取股票代码作为列名
stock_codes = lasso_data.columns

# 初始化存储回归结果的字典
regression_results = {}

for stock_code in stock_codes:
    # 获取当前股票的预测收益
    lasso_pred = lasso_data[stock_code]
    bmk_pred = bmk_data[stock_code]

    # 创建一个时间序列
    times = pd.date_range(start='08:31:00', periods=len(lasso_pred), freq='T').time

    # 计算LASSO预测值的均值和标准差
    lasso_mean = lasso_pred.mean()
    lasso_std = lasso_pred.std()

    # 计算基准模型预测值的均值和标准差
    bmk_mean = bmk_pred.mean()
    bmk_std = bmk_pred.std()

    # 标准化LASSO和基准模型的预测值
    lasso_standardized = (lasso_pred - lasso_mean) / lasso_std
    bmk_standardized = (bmk_pred - bmk_mean) / bmk_std

    # 构建回归模型
    X = np.column_stack((lasso_standardized, bmk_standardized))
    y = lasso_pred.shift(-1).dropna().values  # 滞后收益作为实际收益率
    X = X[:-1]  # 对齐时间

    # 处理 NaN 值，删除包含 NaN 的样本
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    bmk_standardized_filtered = bmk_standardized[:-1][mask]  # 对基准模型的预测值应用相同的掩码

    # 检查X和y是否为空
    if X.shape[0] > 0 and y.shape[0] > 0:
        # 回归分析
        model = LinearRegression().fit(X, y)
        predictions = model.predict(X)

        # 计算R²和增量R²
        R2_both = r2_score(y, predictions)
        R2_bmk = r2_score(y, bmk_standardized_filtered)  # 使用过滤后的基准模型预测值
        delta_R2 = R2_both - R2_bmk

        # 保存结果
        regression_results[stock_code] = {
            'R2_both': R2_both,
            'R2_bmk': R2_bmk,
            'delta_R2': delta_R2,
            'weights': model.coef_
        }

# 将结果转换为DataFrame
results_df = pd.DataFrame.from_dict(regression_results, orient='index')

# 计算投资组合的Sharpe比率
portfolio_returns = results_df['delta_R2'].values
expected_return = portfolio_returns.mean()
portfolio_std = portfolio_returns.std()
risk_free_rate = 0.01  # 假设的无风险收益率
sharpe_ratio = (expected_return - risk_free_rate) / portfolio_std

# 在DataFrame中加入Sharpe比率
results_df['Sharpe_Ratio'] = sharpe_ratio

# 保存结果为CSV文件
results_df.to_csv(output_file_path)

# 输出回归结果和Sharpe比率
print("回归结果已保存至：", output_file_path)
print(results_df.head())


