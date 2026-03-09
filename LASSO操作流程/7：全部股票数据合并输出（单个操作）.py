#将文件中所有的股票合并成一个csv文件，为滚动预测与基准模型预测提供已经处理好的数据
import pandas as pd
import os

# 定义输入文件夹路径和输出文件路径
input_folder = "C:\\Users\\A\\Desktop\\160105\\0105Retail data"# 输入文件夹路径
output_file = "C:\\Users\\A\\Desktop\\160105\\0105Retail data\\Retail_merged0105.csv"  # 输出文件路径

# 初始化一个空的DataFrame，用于存储合并后的数据
combined_df = pd.DataFrame()

# 遍历输入文件夹中的所有CSV文件
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        # 生成文件的完整路径
        file_path = os.path.join(input_folder, filename)

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 选择文件中的前三列
        selected_columns = df.iloc[:, :2]

        # 将列名更改为以文件名命名，避免重复
        selected_columns.columns = [f'{filename}_col1', f'{filename}_col2']

        # 将选定的三列数据合并到总的DataFrame中
        if combined_df.empty:
            combined_df = selected_columns
        else:
            combined_df = pd.concat([combined_df, selected_columns], axis=1)

# 保存合并后的数据到一个新的CSV文件
combined_df.to_csv(output_file, index=False)

print(f"文件已合并并保存到 {output_file}")
