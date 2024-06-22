import pandas as pd
import numpy as np
from pymfe.mfe import MFE
from sklearn.model_selection import train_test_split

# 加载数据集
file_path = 'winequality-red_set.csv'  # 请确保文件路径正确
data = pd.read_csv(file_path)

# 检查数据结构
print(data.head())

# 准备数据（假设最后一列为目标变量）
X = data.iloc[:, :-1].values  # 去掉最后一列
y = data.iloc[:, -1].values  # 最后一列作为目标变量

# 分割数据集为5个子集
X_splits = []
y_splits = []
for _ in range(5):
    X_part, X, y_part, y = train_test_split(X, y, test_size=0.8, random_state=None)
    X_splits.append(X_part)
    y_splits.append(y_part)

# 提取元特征并保存到CSV
meta_features_list = []

for i, (X_subset, y_subset) in enumerate(zip(X_splits, y_splits)):
    mfe = MFE()
    mfe.fit(X_subset, y_subset)
    ft_names, ft_values = mfe.extract()
    ft_values = [f"wine_{i+1}"] + list(ft_values)  # 添加数据集标识
    meta_features_list.append(ft_values)

# 创建一个DataFrame来保存所有子集的元特征
columns = ['Dataset'] + ft_names
meta_features_df = pd.DataFrame(meta_features_list, columns=columns)

# 保存到CSV文件
output_file_path = 'extracted_redwine.csv'
meta_features_df.to_csv(output_file_path, index=False)

print(f"元特征已保存到 {output_file_path}")