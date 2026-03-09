import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 定义文件路径
lasso_file_path = "C:\\Users\\A\\Desktop\\160105\\新output\\predicted_stock_returns.csv"
ar_models = {
    'AR(1)': "C:\\Users\\A\\Desktop\\160105\\新output\\AR_1基准模型预测.csv",
    'AR(2)': "C:\\Users\\A\\Desktop\\160105\\新output\\AR_2基准模型预测.csv",
    'AR(3)': "C:\\Users\\A\\Desktop\\160105\\新output\\AR_3基准模型预测.csv",
    'AR(4)': "C:\\Users\\A\\Desktop\\160105\\新output\\AR_4基准模型预测.csv",
    'AR(5)': "C:\\Users\\A\\Desktop\\160105\\新output\\AR_5基准模型预测.csv",
    'AR(h*)': "C:\\Users\\A\\Desktop\\160105\\新output\\AR_hstar基准模型预测.csv"
}

# 读取并过滤LASSO模型预测数据  根据股票数量进行调整预测值的范围
lasso_data = pd.read_csv(lasso_file_path, usecols=range(150, 225))
time_col = pd.read_csv(lasso_file_path, usecols=[0])

# 合并时间列以过滤时间范围
lasso_data = pd.concat([time_col, lasso_data], axis=1)
lasso_data.columns.values[0] = 'Time'  # 重命名时间列为'Time'

# 转换'Time'列为时间格式
lasso_data['Time'] = pd.to_datetime(lasso_data['Time'], format='%H%M%S')

# 按时间范围过滤数据（从08:31:00到14:58:00）
start_time = pd.to_datetime('083100', format='%H%M%S').time()
end_time = pd.to_datetime('145800', format='%H%M%S').time()
lasso_data = lasso_data[(lasso_data['Time'].dt.time >= start_time) & (lasso_data['Time'].dt.time <= end_time)]

# 初始化字典以存储所有模型的回归结果
all_model_results = {}

# 遍历每个AR模型  根据股票数量确定预测模型
for model_name, model_path in ar_models.items():
    # 读取并过滤AR模型预测数据
    try:
        ar_data = pd.read_csv(model_path, usecols=range(150, 225))
        ar_time_col = pd.read_csv(model_path, usecols=[0])

        # 合并时间列以过滤时间范围
        ar_data = pd.concat([ar_time_col, ar_data], axis=1)
        ar_data.columns.values[0] = 'Time'  # 重命名时间列为'Time'

        # 转换'Time'列为时间格式
        ar_data['Time'] = pd.to_datetime(ar_data['Time'], format='%H%M%S')

        # 按时间范围过滤数据
        ar_data = ar_data[(ar_data['Time'].dt.time >= start_time) & (ar_data['Time'].dt.time <= end_time)]

    except FileNotFoundError:
        print(f"File for {model_name} not found. Please upload the file.")
        continue

    # 提取股票代码作为列名（排除时间列）
    stock_codes = lasso_data.columns[1:]

    # 初始化字典以存储回归结果和权重
    regression_results = {}
    weights = []

    # 遍历每个股票代码进行回归分析
    for stock_code in stock_codes:
        # 从LASSO和AR模型中获取当前股票的预测收益
        lasso_pred = lasso_data[stock_code]
        ar_pred = ar_data[stock_code]

        # 标准化LASSO和AR模型的预测值
        lasso_standardized = (lasso_pred - lasso_pred.mean()) / lasso_pred.std()
        ar_standardized = (ar_pred - ar_pred.mean()) / ar_pred.std()

        # 构建回归模型的输入和输出
        X = np.column_stack((lasso_standardized, ar_standardized))
        y = lasso_pred.shift(-1).dropna().values  # 滞后收益作为实际收益率

        # 对齐X和y的长度：截断X以匹配y的长度
        X = X[:len(y)]

        # 处理NaN值，删除包含NaN的样本
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        ar_standardized_filtered = ar_standardized[:-1][mask]

        # 检查X和y是否为空
        if X.shape[0] > 0 and y.shape[0] > 0:
            # 执行回归分析
            model = LinearRegression().fit(X, y)
            predictions = model.predict(X)

            # 计算R²和增量R²
            R2_both = r2_score(y, predictions)
            R2_ar = r2_score(y, ar_standardized_filtered)
            delta_R2 = R2_both - R2_ar

            # 存储结果
            regression_results[stock_code] = {
                'R2_both': R2_both,
                'R2_ar': R2_ar,
                'delta_R2': delta_R2,
                'weights': model.coef_
            }

            # 存储系数信息
            weights.append({
                'Stock': stock_code,
                'LASSO_Coefficient': model.coef_[0],
                'AR_Coefficient': model.coef_[1]
            })

        # 将结果转换为DataFrame
        if regression_results:  # 确保字典不为空
            regression_results_df = pd.DataFrame.from_dict(regression_results, orient='index')
            weights_df = pd.DataFrame(weights)

            # 将文件名中的非法字符替换或删除
            safe_model_name = model_name.replace("(", "").replace(")", "").replace("*", "_")

            # 将每个模型的回归结果保存为CSV文件
            regression_results_df.to_csv(f'C:\\Users\\A\\Desktop\\160105\\新output\\{safe_model_name}_regression_results.csv')

    # 将结果存储在字典中以便进一步使用
    all_model_results[model_name] = (regression_results_df, weights_df)

# 对生成的CSV文件中的数据进行描述性分析并保存结果

# 计算每个模型回归结果的R-square均值及其余描述性信息
regression_description = regression_results_df[['R2_both', 'R2_ar', 'delta_R2']].describe()
weights_description = weights_df.describe()

# 初始化列表以存储每个模型的R2_both均值
r2_both_means = []

# 遍历所有模型以计算R2_both的均值
for model_name, (results_df, weights_df) in all_model_results.items():
    # 计算当前模型的R2_both均值
    r2_both_mean = results_df['R2_both'].mean()

    # 将模型名称和均值添加到列表中
    r2_both_means.append({'Model': model_name, 'R2_both_mean': r2_both_mean})

# 创建包含所有模型R2_both均值的DataFrame
r2_both_means_df = pd.DataFrame(r2_both_means)

# 保存所有模型的R2_both均值到一个CSV文件
r2_both_means_df.to_csv('C:\\Users\\A\\Desktop\\160105\\新output\\all_models_R2_both_means.csv', index=False)

# 显示第一个模型的回归结果摘要
for model_name, (results_df, weights_df) in all_model_results.items():
    print(f"Model: {model_name}")
    print("Regression Results:")
    print(results_df.head())
    print("Regression Weights:")
    print(weights_df.head())
    print("\n")
