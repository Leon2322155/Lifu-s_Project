import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 加载原始数据集
original_file_path = 'combined_meta_features_with_metrics.csv'
original_df = pd.read_csv(original_file_path)

# 准备输入特征和标签
X = original_df.iloc[:, 1:-4]  # 除开第一列和最后四列
y = original_df[['CNN_Accuracy', 'CNN_F1', 'DT_Accuracy', 'DT_F1']]

# 切分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化XGBoost回归模型
xgb_model = XGBRegressor(objective='reg:squarederror')

# 使用MultiOutputRegressor处理多输出
multi_output_model = MultiOutputRegressor(xgb_model)

# 训练模型
multi_output_model.fit(X_train, y_train)

# 保存模型
joblib.dump(multi_output_model, 'multi_output_xgb_model.joblib')

# 评估模型
y_pred = multi_output_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
rmse = mean_squared_error(y_test, y_pred, multioutput='raw_values', squared=False)
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
# print("R2 Score (Accuracy):", r2)
