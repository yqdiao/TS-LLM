import pandas as pd
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import numpy as np
import os


def load_and_prepare_data(stock_file_path, market_file_path):
    """
    加载并准备股票数据和市场数据
    """
    # 读取数据
    stock_data = pd.read_csv(stock_file_path)
    market_data = pd.read_csv(market_file_path)

    # 重命名市场因子数据的列名
    market_data_renamed = market_data.copy()
    market_data_renamed.columns = ['time', 'Market_Return']

    # 获取股票数据中第一个时间列的名称
    time_col_name = [col for col in stock_data.columns if 'col1' in col][0]

    # 准备合并
    stock_data_for_merge = stock_data.copy()
    stock_data_for_merge['time'] = stock_data_for_merge[time_col_name]

    # 合并数据
    merged_data = pd.merge(stock_data_for_merge,
                           market_data_renamed,
                           on='time',
                           how='left')

    # 对Market_Return减去0.05%
    merged_data['Market_Return'] = merged_data['Market_Return'] - 0.0005

    return merged_data, time_col_name


def select_best_lag(returns, market_returns, max_lag=5):
    """
    使用BIC准则选择最佳滞后期数
    """
    best_lag = 1
    best_bic = np.inf

    for lag in range(1, max_lag + 1):
        try:
            # 创建滞后变量
            lag_cols = {f'lag{l}': returns.shift(l) for l in range(1, lag + 1)}
            df = pd.DataFrame(lag_cols)
            df['y'] = returns
            df['market'] = market_returns
            df = df.dropna()

            if len(df) > lag + 1:  # 确保有足够的数据
                X = df[[f'lag{l}' for l in range(1, lag + 1)] + ['market']]
                y = df['y']
                X = sm.add_constant(X)

                model = OLS(y, X).fit()
                if model.bic < best_bic:
                    best_bic = model.bic
                    best_lag = lag
        except Exception as e:
            print(f"计算滞后期 {lag} 时出错: {str(e)}")
            continue

    return best_lag


def forecast_mixed_model(stock_data, time_col_name, lag, model_type, n_stocks):
    """
    使用混合模型预测股票收益率
    """
    prediction_columns = {}

    for i in range(n_stocks):
        # 获取当前股票数据
        returns_col = stock_data.iloc[:, 2 * i + 1]
        market_returns = stock_data['Market_Return']

        # 删除缺失值
        valid_mask = returns_col.notna() & market_returns.notna()
        returns_col = returns_col[valid_mask]
        market_returns = market_returns[valid_mask]

        # 初始化预测结果
        forecast = np.full(len(stock_data), np.nan)

        if len(returns_col) > lag:
            # 创建滞后变量
            lag_cols = {f'lag{l}': returns_col.shift(l) for l in range(1, lag + 1)}
            df = pd.DataFrame(lag_cols)
            df['y'] = returns_col
            df['market'] = market_returns
            df = df.dropna()

            if len(df) > 0:
                # 准备变量
                X = df[[f'lag{l}' for l in range(1, lag + 1)] + ['market']]
                y = df['y']
                X = sm.add_constant(X)

                # 拟合模型
                model = OLS(y, X).fit()
                params = model.params

                # 进行预测
                for t in range(lag, len(returns_col)):
                    if all(~np.isnan(returns_col.iloc[t - l]) for l in range(1, lag + 1)) and \
                            not np.isnan(market_returns.iloc[t]):
                        pred_x = [1.0]  # 常数项
                        for l in range(1, lag + 1):
                            pred_x.append(returns_col.iloc[t - l])
                        pred_x.append(market_returns.iloc[t])
                        forecast[valid_mask.index[t]] = np.dot(pred_x, params)

        # 保存预测结果
        column_name = f"Stock_{i + 1}_Prediction_{model_type}"
        prediction_columns[column_name] = forecast

    return prediction_columns


def main():
    # 文件路径
    stock_file_path = "C:\\Users\\A\\Desktop\\160105\\selected_merged_file0105.csv"
    market_file_path = "C:\\Users\\A\\Desktop\\160105\\Nikkei 225\\Nikkei225.csv"
    output_folder = "C:\\Users\\A\\Desktop\\160105\\LASSO操作流程\\混合模型预测结果"

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 加载和准备数据
    stock_data, time_col_name = load_and_prepare_data(stock_file_path, market_file_path)

    # 计算股票数量
    n_stocks = (len(stock_data.columns) - 2) // 2  # 减去time和Market_Return列

    # 定义模型名称
    model_names = ["AR(1)", "AR(2)", "AR(3)", "AR(4)", "AR(5)", "AR(h*)"]

    # 对每个模型进行预测
    for model_name in model_names:
        print(f"正在处理模型: {model_name}")

        if model_name == "AR(h*)":
            # 对第一只股票数据进行示例拟合以选择最佳滞后期
            returns = stock_data.iloc[:, 1]
            market_returns = stock_data['Market_Return']
            best_lag = select_best_lag(returns, market_returns)
            lag = best_lag
            # 修改保存文件名，将 AR(h*) 改为 AR_h_star
            save_model_name = "AR_h_star"
        else:
            lag = int(model_name.split("(")[1].split(")")[0])
            save_model_name = model_name.replace("(", "_").replace(")", "")

        # 进行预测
        predictions = forecast_mixed_model(stock_data, time_col_name, lag, model_name, n_stocks)

        # 准备输出数据
        output_data = stock_data.copy()
        for col_name, pred_values in predictions.items():
            output_data[col_name] = pred_values

        # 删除083000之前的预测值
        for col in output_data.columns:
            if 'Prediction' in col:
                output_data.loc[output_data[time_col_name] < 83000, col] = np.nan

        # 使用修改后的文件名保存结果
        output_file = os.path.join(output_folder, f"{save_model_name}_Market_Mixed_Model_Prediction.csv")
        output_data.to_csv(output_file, index=False)
        print(f"模型 {model_name} 的预测结果已保存到: {output_file}")


if __name__ == "__main__":
    main()