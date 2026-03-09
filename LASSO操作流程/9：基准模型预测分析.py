#基于基准模型预测出的数据进一步进行拟合，然后统计模型中的系数以及拟合度
import pandas as pd
import statsmodels.api as sm
import numpy as np

# 设置文件路径模板（请修改为你的本地路径）
file_path_template = "C:\\Users\\A\\Desktop\\160105\\retail output\\{model_name}基准模型预测.csv"
output_file_path_template = "C:\\Users\\A\\Desktop\\160105\\retail output\\{model_name}系数描述性统计.csv"

# 定义模型名称列表
model_names = ["AR_1", "AR_2", "AR_3", "AR_4", "AR_5", "AR_hstar"]

# 初始化列表以存储结果
results = []

# 对每个模型执行回归分析
for model_name in model_names:
    # 设置输入和输出文件路径
    file_path = file_path_template.format(model_name=model_name)
    output_file_path = output_file_path_template.format(model_name=model_name)

    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 过滤掉083000时刻及之前的数据行
    data = data[data.iloc[:, 0] > 83000]

    # 打印数据的前几行，检查数据是否正确读取
    print(f"Processing {model_name}:")
    print(data.head())

    # 初始化列表以存储回归分析结果
    stock_codes = []
    alpha_coefficients = []
    gamma_coefficients = []
    r_squared_values = []

    # 处理数据，从第201列到301列，针对每只股票进行回归分析
    for i in range(150, 225):
        # 获取股票预测收益率列名
        stock_code = data.columns[i]

        # 获取对应的滞后收益率
        lagged_return = data.iloc[:, i - 100]

        # 获取预测收益率
        predicted_lagged_return_bmk = data.iloc[:, i]

        # 将预测收益率中 '150000' 的缺失值用0代替
        predicted_lagged_return_bmk.fillna(0, inplace=True)

        # 删除滞后收益率和预测收益率中的其他NaN值
        combined_data = pd.concat([lagged_return, predicted_lagged_return_bmk], axis=1).dropna()
        if combined_data.empty:
            continue

        lagged_return = combined_data.iloc[:, 0]
        predicted_lagged_return_bmk = combined_data.iloc[:, 1]

        # 标准化预测滞后收益率
        normalized_prediction_bmk = (
            predicted_lagged_return_bmk - predicted_lagged_return_bmk.mean()) / predicted_lagged_return_bmk.std()
        normalized_prediction_bmk.replace([np.inf, -np.inf], np.nan, inplace=True)
        normalized_prediction_bmk.dropna(inplace=True)

        # 确保数据对齐且没有NaN值
        lagged_return = lagged_return.loc[normalized_prediction_bmk.index]
        normalized_prediction_bmk = normalized_prediction_bmk.loc[lagged_return.index]

        if lagged_return.empty or normalized_prediction_bmk.empty:
            continue

        # 为模型添加常量
        X = sm.add_constant(normalized_prediction_bmk, has_constant='add')
        y = lagged_return

        # 拟合回归模型
        model = sm.OLS(y, X).fit()

        # 提取系数
        alpha = model.params.get('const', np.nan)
        gamma = model.params.get(normalized_prediction_bmk.name, np.nan)
        r_squared = model.rsquared

        # 将结果添加到列表
        stock_codes.append(stock_code)
        alpha_coefficients.append(alpha)
        gamma_coefficients.append(gamma)
        r_squared_values.append(r_squared)

    # 创建包含结果的DataFrame
    regression_results = pd.DataFrame({
        'Stock Code': stock_codes,
        'Alpha Coefficient': alpha_coefficients,
        'Gamma Coefficient': gamma_coefficients,
        'R-Squared': r_squared_values
    })

    # 对系数进行描述性统计
    descriptive_stats = regression_results.describe()

    # 合并结果和描述性统计数据到一个DataFrame中
    combined_results = pd.concat([regression_results, descriptive_stats.T], axis=0)

    # 将回归系数和描述性统计保存到CSV文件
    combined_results.to_csv(output_file_path, index=True)

    # 打印描述性统计结果
    print(f"{model_name} model regression results:")
    print(descriptive_stats)
    print(f"Results saved to {output_file_path}\n")

    # 将结果添加到总结果列表
    results.append(combined_results)

# 打印所有模型的总结果
print("All models processed.")
