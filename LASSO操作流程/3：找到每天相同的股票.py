#找到与第一天处理的相同的股票，以2016年1月4日选出的股票为基准
import os
import shutil
import re

# Define the file paths
source_path = "E:\\stock and lag return\\测试数据"
output_path = "E:\\stock and lag return\\201601output\\0105股票"
destination_path = "E:\\stock and lag return\\0105测试数据"

# Function to extract the numeric part from the file name
def extract_numeric(filename):
    match = re.search(r'\d+', filename)
    return match.group() if match else None

# List all CSV files in the source directory and extract numeric parts
source_files_numeric = {extract_numeric(f) for f in os.listdir(source_path) if f.endswith('.csv')}
print(f"Total unique stock codes in source directory: {len(source_files_numeric)}")  # Debugging line
print(f"Source stock codes: {source_files_numeric}")  # Debugging line

# Ensure the destination path exists
os.makedirs(destination_path, exist_ok=True)

# List all CSV files in the output directory, extract numeric parts, and filter those that match source files
output_files = [f for f in os.listdir(output_path) if f.endswith('.csv') and extract_numeric(f) in source_files_numeric]
print(f"Matching CSV files found in output directory: {len(output_files)}")  # Debugging line
print(f"Output files: {output_files}")  # Debugging line

# Copy matching files to the destination directory
for file_name in output_files:
    shutil.copy(os.path.join(output_path, file_name), os.path.join(destination_path, file_name))

print(f"{len(output_files)} matching files have been copied to the destination folder.")



