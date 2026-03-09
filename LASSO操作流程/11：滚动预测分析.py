#基于滚动预测出的数据进一步进行拟合，然后统计模型中的系数以及拟合度
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load the CSV file from the provided path
file_path = "C:\\Users\\A\\Desktop\\160105\\新output\\predicted_stock_returns.csv"
output_file_path = "C:\\Users\\A\\Desktop\\160105\\新output\\LASSO系数分析.csv"

# Read the CSV file, only columns 200 to 299 根据股票的数量随时调整
data = pd.read_csv(file_path, usecols=range(150, 225))

# Initialize lists to store the results
stock_codes = []
alpha_coefficients = []
beta_coefficients = []
r_squared_values = []

# Process data with index bounds check to avoid out-of-bounds errors
for i in range(0, data.shape[1], 3):
    if i + 1 >= data.shape[1]:  # Ensure i+1 is within bounds
        break

    stock_code = data.columns[i]
    lagged_return = data.iloc[:, i]
    predicted_lagged_return = data.iloc[:, i + 1]

    # Remove NaN values
    combined_data = pd.concat([lagged_return, predicted_lagged_return], axis=1).dropna()
    if combined_data.empty:
        continue

    lagged_return = combined_data.iloc[:, 0]
    predicted_lagged_return = combined_data.iloc[:, 1]

    # Normalize predicted lagged returns
    normalized_prediction = (predicted_lagged_return - predicted_lagged_return.mean()) / predicted_lagged_return.std()
    normalized_prediction.replace([np.inf, -np.inf], np.nan, inplace=True)
    normalized_prediction.dropna(inplace=True)

    # Align data and ensure no NaN values
    lagged_return = lagged_return.loc[normalized_prediction.index]
    normalized_prediction = normalized_prediction.loc[lagged_return.index]

    if lagged_return.empty or normalized_prediction.empty:
        continue

    # Add constant to the model
    X = sm.add_constant(normalized_prediction)
    y = lagged_return

    # Fit regression model
    model = sm.OLS(y, X).fit()

    # Extract coefficients
    alpha = model.params['const']
    beta = model.params[normalized_prediction.name]
    r_squared = model.rsquared

    # Append results to lists
    stock_codes.append(stock_code)
    alpha_coefficients.append(alpha)
    beta_coefficients.append(beta)
    r_squared_values.append(r_squared)

# Create a DataFrame with the results
regression_results = pd.DataFrame({
    'Stock Code': stock_codes,
    'Alpha Coefficient': alpha_coefficients,
    'Beta Coefficient': beta_coefficients,
    'R-Squared': r_squared_values
})

# Perform descriptive statistics on the coefficients
descriptive_stats = regression_results.describe()

# Combine the results and descriptive statistics into one DataFrame
combined_results = pd.concat([regression_results, descriptive_stats.T], axis=0)

# Save the regression coefficients and descriptive statistics to a CSV file
combined_results.to_csv(output_file_path, index=True)

# Display descriptive statistics
print(descriptive_stats)
