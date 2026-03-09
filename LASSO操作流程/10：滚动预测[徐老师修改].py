#滚动窗口的设置：我们使用过去30分钟的数据来预测下一分钟的收益率。
# LASSO模型的应用：使用LASSO模型来选择重要的滞后收益率作为预测变量，并预测下一分钟的收益率。
#缺少K=10 缺少交叉验证来寻找最优误差系数(cv.glmnet用此库函数)
#不需要预测交易日最后一分钟的滞后收益率
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV


# 加载数据
file_path = "C:\\Users\\A\\Desktop\\LASSO操作流程\\selected_merged0105.csv"
data = pd.read_csv(file_path)

# 股票数量
n_stocks = data.shape[1] // 2

# 初始化预测结果存储
new_data = data.copy()

# 保存所有预测结果的字典
prediction_columns = {}
best_lambdas = {}  # 用于存储每只股票每个窗口的最优alpha

# 对每只股票进行预测
for i in range(n_stocks):
    # 获取当前股票的时间和滞后收益率
    time_col_name = data.columns[2 * i]  # 获取股票交易时间的列名作为股票代码
    returns_col = data.iloc[:, 2 * i + 1]

    # 删除缺失值
    valid_idx = returns_col.dropna().index
    returns_col = returns_col.dropna()

    # 标准化收益率
    scaler = StandardScaler()
    returns_col_scaled = scaler.fit_transform(returns_col.values.reshape(-1, 1)).flatten()

    # 滚动窗口预测
    window_size = 30  # 30分钟窗口
    forecast = []
    lambda_per_stock = []  # 保存每只股票的最优alpha值

    # 数据起点的索引调整，从080100时刻开始，假设数据从080000时刻开始计数，且080000的值为NaN
    start_index = 2  # 因为080000为NaN，从080100时刻开始
    # 假设已经有了经过预处理的收益率数据序列returns_col_scaled
    # 滚动窗口预测
    # ============================以下为修改==================================================================
    # 初始化用于存储所有训练样本特征的X_train和目标变量的y_train
    X_train = []
    y_train = []
    
    for j in range(start_index + window_size, len(returns_col_scaled)):
        # 确保不预测最后一个时间点（可根据实际需求决定是否保留这一判断，比如如果想预测最后一个点可以去掉这行）
        if j == len(returns_col_scaled) - 1:
            continue
    
        # 准备当前窗口的训练数据，使用前30分钟的数据作为特征（X_train）来预测下一分钟的数据（y_train）
        current_X_train = np.array([returns_col_scaled[j - window_size:j]]).reshape(1, -1)  # 重塑形状为 (1, 30)，符合模型输入特征的形状要求（假设模型期望二维输入，一行代表一个样本，这里每次只有一个样本，30列代表30个特征，即前30分钟的数据）
        current_y_train = np.array([returns_col_scaled[j]]).reshape(1, -1)  # 将下一分钟的数据作为目标变量，同样重塑形状为 (1, 1)，符合目标变量的形状要求，即每次预测一个值
    
        # 将当前窗口的样本数据添加到总的训练数据列表中
        X_train.append(current_X_train[0])  # 取current_X_train的第一行（因为其形状为(1, 30)）添加到X_train列表中
        y_train.append(current_y_train[0])  # 取current_y_train的第一行（因为其形状为(1, 1)）添加到y_train列表中
    
    # 将列表转换为numpy数组，并调整形状（这部分代码要放在循环外面，避免重复转换以及错误使用append方法）
    X_train = np.array(X_train).reshape(-1, window_size)
    y_train = np.array(y_train).reshape(-1, 1)
    
    # 最终确保X_train形状为 (n_samples, n_features)，y_train的形状为（n_samples,1）
    # =======================================================修改结束==========================================

    # # 预测开始的索引为31（表示从083100时刻开始预测）
    # for j in range(start_index + window_size, len(returns_col_scaled)):
    #     # 确保不预测最后一个时间点
    #     if j == len(returns_col_scaled) - 1:
    #         continue

    #     # 准备当前窗口的训练数据
    #     X_train = np.array([returns_col_scaled[j - window_size:j]])
    #     y_train = returns_col_scaled[j - window_size + 1:j + 1]      
        

        # 使用LassoCV进行10折交叉验证，增加max_iter和tol
    lasso_cv = LassoCV(cv=10, max_iter=50000, tol=0.001)
    lasso_cv.fit(X_train, y_train)  # 使用X_train.T使其形状为 (n_samples, n_features)
        
        # 获取最优的alpha值
    best_alpha = lasso_cv.alpha_
    lambda_per_stock.append(best_alpha)

    # 使用最优的alpha进行预测
    if j + 1 < len(returns_col_scaled):
        X_test = np.array([returns_col_scaled[j - window_size + 1:j + 1]])
        pred = lasso_cv.predict(X_test.T)
        forecast.append(pred[0])

    # 将预测结果与valid_idx对齐
    forecast_full = [np.nan] * (window_size + start_index) + forecast  # 确保第一个预测值从083100时刻开始

    # 确保预测长度与new_data对齐
    forecast_full = forecast_full[:len(new_data)]

    # 使用股票代码作为预测结果列名
    prediction_columns[time_col_name] = forecast_full
    best_lambdas[time_col_name] = lambda_per_stock  # 保存每只股票的最优alpha值

# 使用pd.concat一次性将预测结果添加到DataFrame中
new_columns_df = pd.DataFrame(prediction_columns)
new_data = pd.concat([new_data, new_columns_df], axis=1)

# 复制DataFrame以消除碎片化
new_data = new_data.copy()

# 将预测结果保存为CSV文件
output_file_path = "C:\\Users\\A\\Desktop\\LASSO操作流程\\predicted_stock_returns.csv"
new_data.to_csv(output_file_path, index=False)

# 创建一个DataFrame来存储每只股票的最优alpha值
best_lambdas_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in best_lambdas.items()]))

# 将最优alpha值保存为CSV文件
lambda_output_file_path = "C:\\Users\\A\\Desktop\\LASSO操作流程\\best_lambdas.csv"
best_lambdas_df.to_csv(lambda_output_file_path, index=False)

# 输出前几行检查
print(new_data.head())

# 输出每只股票的最优alpha值
for stock, lambdas in best_lambdas.items():
    print(f"{stock}: 最优alpha值 {lambdas}")
