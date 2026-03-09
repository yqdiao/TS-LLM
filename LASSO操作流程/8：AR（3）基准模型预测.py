#用AR（3）基准模型预测每只股票的每分钟收益率
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

# 加载数据
file_path = "E:\\stock and lag return\\merged_file.csv"
data = pd.read_csv(file_path)

# 提取股票数目
n_stocks = data.shape[1] // 2  # 每只股票有两列数据

# 初始化预测结果存储的字典
prediction_columns = {}

# 对每只股票进行预测
for i in range(n_stocks):
    # 获取当前股票的时间和滞后收益率
    time_col = data.iloc[:, 2 * i]
    returns_col = data.iloc[:, 2 * i + 1]

    # 删除缺失值
    valid_idx = returns_col.dropna().index
    returns_col = returns_col.dropna()

    # 构建AR(3)模型并进行滚动预测
    forecast = [np.nan] * 3  # 前3个数据点没有预测值

    model = AutoReg(returns_col, lags=3)
    model_fit = model.fit()

    predictions = model_fit.predict(start=3, end=len(returns_col) - 1, dynamic=False)
    forecast.extend(predictions)

    # 预测结果长度与原始数据对齐
    forecast_full = [np.nan] * (len(data) - len(forecast)) + forecast

    # 保存预测结果到字典
    new_column_name = f"Stock_{i + 1}_Prediction"
    prediction_columns[new_column_name] = forecast_full

# 将预测结果插入到原始DataFrame中每只股票的滞后收益列之后
for i in range(n_stocks):
    new_column_name = f"Stock_{i + 1}_Prediction"
    data.insert(2 * i + 2 + i, new_column_name, prediction_columns[new_column_name])

# 删除预测列中083000之前的预测值，只保留083000及之后的预测数据
for i in range(n_stocks):
    prediction_col_name = f"Stock_{i + 1}_Prediction"
    data.loc[data.iloc[:, 0] < 83000, prediction_col_name] = np.nan

# 复制 DataFrame 以消除碎片化
data = data.copy()

# 保存结果为CSV文件
output_file_path = "E:\\stock and lag return\\基准模型预测.csv"
data.to_csv(output_file_path, index=False)

# 输出前几行检查
print(data.head())


