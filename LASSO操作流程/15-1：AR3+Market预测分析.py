import pandas as pd
import statsmodels.api as sm
import numpy as np

# 设置文件路径
file_path = "C:\\Users\\A\\Desktop\\160105\\AR3和Market混合模型预测.csv"
output_file_path = "C:\\Users\\A\\Desktop\\160105\\AR3和Market混合模型预测分析.csv"

# 读取CSV文件
data = pd.read_csv(file_path)

print(f"Processing the file: {file_path}")
print(data.head())

# 过滤掉083000时刻及之前的数据行
data = data[data.iloc[:, 0] > 83000]

# 初始化存储结果的列表
stock_codes = []
alpha_coefficients = []
gamma_coefficients = []
r_squared_values = []

# 计算股票数量
n_stocks = (data.shape[1] - 1) // 3

# 对每只股票执行回归分析
for i in range(n_stocks):
    stock_code = f"Stock_{i + 1}"

    # 获取该股票的列索引
    time_col = 3 * i + 1
    lagged_col = 3 * i + 2
    predicted_col = 3 * i + 3

    # 提取数据，确保获取整列数据而不是单个值
    lagged_return = data.iloc[:, lagged_col]
    predicted_lagged_return = data.iloc[:, predicted_col]

    # 数据预处理
    # 删除任何包含NaN的行
    valid_data = pd.DataFrame({
        'lagged': lagged_return,
        'predicted': predicted_lagged_return
    }).dropna()

    if len(valid_data) < 2:  # 确保有足够的数据点进行回归
        continue

    # 标准化预测值
    predicted_normalized = (valid_data['predicted'] - valid_data['predicted'].mean()) / valid_data['predicted'].std()

    # 准备回归数据
    X = sm.add_constant(predicted_normalized)
    y = valid_data['lagged']

    try:
        # 拟合回归模型
        model = sm.OLS(y, X).fit()

        # 提取系数和R方
        alpha = model.params['const']
        gamma = model.params[1]  # 第二个系数
        r_squared = model.rsquared

        # 存储结果
        stock_codes.append(stock_code)
        alpha_coefficients.append(alpha)
        gamma_coefficients.append(gamma)
        r_squared_values.append(r_squared)

    except Exception as e:
        print(f"Error processing {stock_code}: {str(e)}")
        continue

# 创建结果DataFrame
regression_results = pd.DataFrame({
    'Stock Code': stock_codes,
    'Alpha Coefficient': alpha_coefficients,
    'Gamma Coefficient': gamma_coefficients,
    'R-Squared': r_squared_values
})

# 计算描述性统计
descriptive_stats = regression_results.describe()

# 合并结果
combined_results = pd.concat([regression_results, descriptive_stats], axis=0)

# 保存结果
combined_results.to_csv(output_file_path, index=True)

print(f"Regression results for the file {file_path}:")
print(descriptive_stats)
print(f"Results saved to {output_file_path}\n")