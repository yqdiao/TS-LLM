import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Configure output directory
OUTPUT_DIR = r"C:\Users\A\Desktop\160105\retail output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. Load and extract return data
def load_and_extract_returns(file_path):
    """Load CSV and extract only return columns (col2)"""
    df = pd.read_csv(file_path)

    # Keep only return columns (col2)
    return_cols = [col for col in df.columns if 'col2' in col]
    returns_df = df[return_cols]

    # Ensure data is numeric
    for col in returns_df.columns:
        if returns_df[col].dtype == 'object':
            returns_df[col] = pd.to_numeric(returns_df[col], errors='coerce')

    # Handle missing values
    returns_df = returns_df.fillna(method='ffill').fillna(0)

    print(f"Original returns data shape: {returns_df.shape}")
    print(f"Number of stocks: {len(returns_df.columns)}")

    return returns_df


# 2. Prepare flattened rolling window data
def prepare_flattened_windows(returns_df, window_size=30, verbose=True):
    """Prepare flattened rolling window data

    Args:
        returns_df: DataFrame containing only return columns
        window_size: Size of window, default 30 minutes
        verbose: Whether to print detailed information

    Returns:
        X_flat_df: Flattened feature DataFrame (each row is a flattened time window)
        y_df: Target variable DataFrame (each row is the next time point's returns for all stocks)
    """
    n_stocks = returns_df.shape[1]

    # Create empty lists to store flattened windows and targets
    X_flat = []
    y_next = []

    # Store window indices for later analysis
    window_indices = []

    # Start from window_size to ensure enough historical data
    for i in range(window_size, len(returns_df)):
        # Current window data for all stocks
        window_data = returns_df.iloc[i - window_size:i].values  # Shape: [window_size, n_stocks]

        # Flatten window data to 1D vector
        flat_window = window_data.flatten()  # Shape: [window_size * n_stocks]

        # Next time point's returns as target
        next_returns = returns_df.iloc[i].values  # Shape: [n_stocks]

        X_flat.append(flat_window)
        y_next.append(next_returns)
        window_indices.append(i)

    # Convert to NumPy arrays
    X_flat_array = np.array(X_flat)
    y_array = np.array(y_next)

    if verbose:
        print(f"\nFlattened training data shape X: {X_flat_array.shape}")
        print(f"Target variable shape y: {y_array.shape}")
        print(f"Features per sample: {X_flat_array.shape[1]}")

        # Calculate dimensions before and after flattening
        expected_features = window_size * n_stocks
        print(f"Verification: {window_size} time points × {n_stocks} stocks = {expected_features} features")

    # Create DataFrames for better viewing and understanding
    # Create meaningful column names for features
    feature_cols = []
    for t in range(window_size):
        time_idx = window_size - t  # Reverse order, like t-30, t-29, ...
        for s in range(n_stocks):
            stock_name = returns_df.columns[s].split('_')[0].split('.')[0]  # Extract stock code
            feature_cols.append(f"t-{time_idx}_{stock_name}")

    # Create DataFrame
    X_flat_df = pd.DataFrame(X_flat_array, columns=feature_cols, index=window_indices)

    # Create column names for target variables
    target_cols = [f"next_{col.split('_')[0].split('.')[0]}" for col in returns_df.columns]
    y_df = pd.DataFrame(y_array, columns=target_cols, index=window_indices)

    # Display some data
    if verbose:
        print("\nFirst 5 rows and 5 columns of flattened feature data:")
        print(X_flat_df.iloc[:5, :5])

        print("\nFirst 5 rows and 5 columns of target variable data:")
        print(y_df.iloc[:5, :5])

    return X_flat_df, y_df


# 3. Visualize the first sample's 2D structure
def visualize_first_sample(X_flat_df, returns_df, window_size):
    """Visualize the original 2D structure of the first sample

    Args:
        X_flat_df: Flattened feature DataFrame
        returns_df: Original returns DataFrame
        window_size: Window size
    """
    n_stocks = returns_df.shape[1]

    # Get the first sample and reshape to 2D
    first_sample = X_flat_df.iloc[0].values
    reshaped_sample = first_sample.reshape(window_size, n_stocks)

    # Create reshaped DataFrame for visualization
    stock_names = [col.split('_')[0].split('.')[0] for col in returns_df.columns]
    time_indices = [f"t-{window_size - i}" for i in range(window_size)]

    sample_df = pd.DataFrame(reshaped_sample, index=time_indices, columns=stock_names)

    print("\nFirst sample reshaped to original 2D structure (30 minutes × stocks):")
    print(sample_df.iloc[:5, :5])  # Display first 5 rows and 5 columns

    return sample_df


# 4. Visualize time series for a specific stock
def plot_stock_timeseries(sample_df, stock_idx=0):
    """Plot time series for a specific stock"""
    stock_name = sample_df.columns[stock_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(sample_df.index, sample_df[stock_name], marker='o')
    plt.title(f'30-minute Returns Time Series for Stock {stock_name}', fontsize=14)
    plt.xlabel('Time Lag', fontsize=12)
    plt.ylabel('Returns', fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Save to specified output directory
    output_path = os.path.join(OUTPUT_DIR, f'stock_{stock_name}_timeseries.png')
    plt.savefig(output_path)
    print(f"Chart saved to: {output_path}")
    plt.close()


# 5. Save processed data
def save_processed_data(X_flat_df, y_df, output_prefix='processed_data'):
    """Save processed data for later use"""
    # Ensure using specified output directory
    output_path_X = os.path.join(OUTPUT_DIR, f'{output_prefix}_X.csv')
    output_path_y = os.path.join(OUTPUT_DIR, f'{output_prefix}_y.csv')

    X_flat_df.to_csv(output_path_X)
    y_df.to_csv(output_path_y)
    print(f"\nData saved to:")
    print(f"- {output_path_X}")
    print(f"- {output_path_y}")


# Run all steps
def main():
    # File path - use the provided data path
    file_path = os.path.join(OUTPUT_DIR, "Retail_merged0105.csv")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: Input file not found at {file_path}")
        print(f"Please ensure the file exists or modify the file path")
        return

    print(f"Processing file: {file_path}")
    print(f"Output directory: {OUTPUT_DIR}")

    # 1. Load and extract return data
    returns_df = load_and_extract_returns(file_path)

    # 2. Prepare flattened window data
    window_size = 30
    X_flat_df, y_df = prepare_flattened_windows(returns_df, window_size)

    # 3. Visualize the first sample's 2D structure
    sample_df = visualize_first_sample(X_flat_df, returns_df, window_size)

    # 4. Plot time series for the first stock
    plot_stock_timeseries(sample_df, stock_idx=0)

    # 5. Save processed data
    save_processed_data(X_flat_df, y_df)

    # Output final shape summary
    print("\nFinal data shape summary:")
    print(f"Input features (X): {X_flat_df.shape}")
    print(f"Target variables (y): {y_df.shape}")


# Execute main function
if __name__ == "__main__":
    main()