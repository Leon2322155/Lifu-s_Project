import pandas as pd

# 加载三个数据集
banana_df = pd.read_csv('extracted_meta_features_banana_subsets_with_metrics.csv')
diabetes_df = pd.read_csv('extracted_meta_features_diabetes_subsets_with_metrics.csv')
iris_df = pd.read_csv('extracted_meta_features_iris_subsets_with_metrics.csv')

# # 添加一个列标识数据集来源
# banana_df['Source'] = 'Banana'
# diabetes_df['Source'] = 'Diabetes'
# iris_df['Source'] = 'Iris'

# 合并三个数据集
combined_df = pd.concat([banana_df, diabetes_df, iris_df], ignore_index=True)

# 处理缺失值，可以选择填充或删除
# 这里我们选择填充缺失值，填充方法可以根据需要修改
combined_df.fillna(combined_df.mean(), inplace=True)

# 保存到一个新的CSV文件
output_file_path = 'combined_meta_features_with_metrics.csv'
combined_df.to_csv(output_file_path, index=False)

print(f"三个数据集已整合并处理缺失值，保存到 {output_file_path}")
