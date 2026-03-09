import pandas as pd
import statsmodels.api as sm
import numpy as np
import os

# 设置基础路径
base_folder = "C:\\Users\\A\\Desktop\\160105\\新output"

# 定义模型名称列表
model_names = ["AR_1", "AR_2", "AR_3", "AR_4", "AR_5", "AR_h_star"]

# 确保输出文件夹存在
if not os.path.exists(base_folder):
    os.makedirs(base_folder)

# 处理每个模型的数据文件
for model_name in model_names:
    # 构建输入文件路径
    input_file = os.path.join(base_folder, f"{model_name}_Market_Mixed_Model_Prediction.csv")

    # 构建输出文件路径
    output_file = os.path.join(base_folder, f"{model_name}_Market_Mixed_Model_Prediction_分析.csv")

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        continue

    # 读取CSV文件
    try:
        data = pd.read_csv(input_file)
    except Exception as e:
        print(f"无法读取文件 {input_file}，错误信息：{e}")
        continue

    print(f"正在处理文件: {input_file}")
    print(data.head())

    # 假设第一列为时间列，过滤掉083000时刻及之前的数据行
    time_col = data.columns[0]
    data_filtered = data[data[time_col] > 83000].copy()

    # 初始化存储结果的列表
    stock_codes = []
    alpha_coefficients = []
    gamma_coefficients = []
    r_squared_values = []

    # 计算股票数量，假设每只股票有3列数据：时间、滞后收益率、预测收益率
    n_stocks = (data_filtered.shape[1] - 1) // 3

    # 对每只股票执行回归分析
    for i in range(n_stocks):
        stock_code = f"Stock_{i + 1}"

        # 获取该股票的列索引
        time_col_idx = 3 * i
        lagged_col_idx = 3 * i + 1
        predicted_col_idx = 3 * i + 2

        # 确保列索引不超出范围
        if predicted_col_idx >= data_filtered.shape[1]:
            print(f"警告：文件 {model_name} 中股票 {i + 1} 的列索引超出范围，跳过该股票。")
            continue

        # 提取数据，确保获取整列数据而不是单个值
        lagged_return = data_filtered.iloc[:, lagged_col_idx]
        predicted_lagged_return = data_filtered.iloc[:, predicted_col_idx]

        # 数据预处理：删除任何包含NaN的行
        valid_data = pd.DataFrame({
            'lagged': lagged_return,
            'predicted': predicted_lagged_return
        }).dropna()

        if len(valid_data) < 2:  # 确保有足够的数据点进行回归
            print(f"警告：股票 {stock_code} 的有效数据点少于2，跳过该股票。")
            continue

        # 标准化预测值
        predicted_normalized = (valid_data['predicted'] - valid_data['predicted'].mean()) / valid_data[
            'predicted'].std()

        # 准备回归数据
        X = sm.add_constant(predicted_normalized)
        y = valid_data['lagged']

        try:
            # 拟合回归模型
            model = sm.OLS(y, X).fit()

            # 提取系数和R方
            alpha = model.params['const']
            gamma = model.params['predicted']
            r_squared = model.rsquared

            # 存储结果
            stock_codes.append(stock_code)
            alpha_coefficients.append(alpha)
            gamma_coefficients.append(gamma)
            r_squared_values.append(r_squared)

        except Exception as e:
            print(f"处理股票 {stock_code} 时出错：{e}")
            continue

    # 创建结果DataFrame
    regression_results = pd.DataFrame({
        'Stock Code': stock_codes,
        'Alpha Coefficient': alpha_coefficients,
        'Gamma Coefficient': gamma_coefficients,
        'R-Squared': r_squared_values
    })

    if regression_results.empty:
        print(f"文件 {model_name} 中没有有效的回归结果，跳过保存。")
        continue

    # 计算描述性统计
    descriptive_stats = regression_results.describe()

    # 合并结果和描述性统计数据
    combined_results = pd.concat([regression_results, descriptive_stats], axis=0)

    # 保存结果到CSV文件
    try:
        combined_results.to_csv(output_file, index=True)
        print(f"文件 {input_file} 的回归分析结果已保存到 {output_file}\n")
    except Exception as e:
        print(f"无法保存文件 {output_file}，错误信息：{e}")
        continue

print("所有文件的预测分析已完成。")