import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

# 加载数据
file_path = "C:\\Users\\A\\Desktop\\160105\\retail output\\Retail_merged0105.csv"
data = pd.read_csv(file_path)

# 提取股票数目
n_stocks = data.shape[1] // 2  # 每只股票有两列数据


def forecast_AR_model(lags, model_name):
    """
    用指定的AR模型预测股票收益率
    """
    # 创建一个新的DataFrame来避免列冲突
    data_copy = data.copy()

    # 初始化预测结果存储的字典
    prediction_columns = {}

    # 对每只股票进行预测
    for i in range(n_stocks):
        try:
            # 获取当前股票的时间列名作为股票代码
            stock_code = data_copy.columns[2 * i]
            returns_col = data_copy.iloc[:, 2 * i + 1]

            # 检查数据是否为空
            if returns_col.empty:
                print(f"警告: {stock_code} 的数据为空，跳过此股票")
                continue

            # 删除缺失值前检查数据
            non_nan_count = returns_col.notna().sum()
            if non_nan_count == 0:
                print(f"警告: {stock_code} 的所有数据都是NaN，跳过此股票")
                continue

            # 删除缺失值
            returns_col = returns_col.dropna()

            # 确保还有足够的数据进行建模
            if len(returns_col) <= max(5, lags if lags is not None else 5):
                print(f"警告: {stock_code} 的有效数据点不足，跳过此股票")
                continue

            # 动态选择滞后期数（针对AR(h*)模型）
            if model_name == "AR(h*)":
                best_bic = np.inf
                best_lag = 1
                for lag in range(1, 6):  # 遍历1到5个滞后期
                    try:
                        model = AutoReg(returns_col, lags=lag)
                        model_fit = model.fit()
                        bic = model_fit.bic
                        if bic < best_bic:
                            best_bic = bic
                            best_lag = lag
                    except Exception as e:
                        print(f"警告: 在尝试 {stock_code} 的lag={lag}时发生错误: {str(e)}")
                        continue
                lags = best_lag

            # 构建AR模型并进行滚动预测
            current_lags = lags if lags is not None else 1
            forecast = [np.nan] * current_lags  # 前几个数据点没有预测值

            try:
                model = AutoReg(returns_col, lags=current_lags)
                model_fit = model.fit()

                predictions = model_fit.predict(start=current_lags, end=len(returns_col) - 1, dynamic=False)
                forecast.extend(predictions)

                # 预测结果长度与原始数据对齐
                forecast_full = [np.nan] * (len(data_copy) - len(forecast)) + forecast

                # 使用股票代码作为预测结果列名
                prediction_columns[stock_code] = forecast_full

            except Exception as e:
                print(f"警告: 在处理 {stock_code} 时发生错误: {str(e)}")
                continue

        except Exception as e:
            print(f"警告: 在处理第 {i + 1} 只股票时发生错误: {str(e)}")
            continue

    # 检查是否有成功的预测结果
    if not prediction_columns:
        raise ValueError("没有任何成功的预测结果！请检查输入数据。")

    # 将所有预测结果拼接到一个DataFrame中
    prediction_df = pd.DataFrame(prediction_columns)

    # 合并预测结果到原始DataFrame中
    data_copy = pd.concat([data_copy, prediction_df], axis=1)

    # 复制 DataFrame 以消除碎片化
    data_copy = data_copy.copy()

    # 修改文件名，替换掉不支持的字符
    model_name_safe = model_name.replace('(', '_').replace(')', '').replace('*', 'star')

    # 保存结果为CSV文件
    output_file_path = f"C:\\Users\\A\\Desktop\\160105\\retail output\\{model_name_safe}基准模型预测.csv"
    data_copy.to_csv(output_file_path, index=False)

    # 输出前几行检查
    print(f"{model_name} model predictions saved to {output_file_path}")
    print(data_copy.head())


# 执行不同的AR模型预测并保存结果
try:
    forecast_AR_model(1, "AR(1)")
    forecast_AR_model(2, "AR(2)")
    forecast_AR_model(3, "AR(3)")
    forecast_AR_model(4, "AR(4)")
    forecast_AR_model(5, "AR(5)")
    forecast_AR_model(None, "AR(h*)")
except Exception as e:
    print(f"执行过程中发生错误: {str(e)}")