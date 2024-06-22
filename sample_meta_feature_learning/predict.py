import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 加载训练好的模型
multi_output_model = joblib.load('multi_output_xgb_model.joblib')

# 加载新数据集
new_file_path = 'extracted_redwine.csv'
new_df = pd.read_csv(new_file_path)

# 准备输入特征
X_new = new_df.iloc[:, 1:]  # 使用所有列作为特征

# 进行预测
y_pred_new = multi_output_model.predict(X_new)

# 将预测结果添加到新数据集中
new_df['CNN_Accuracy'] = y_pred_new[:, 0]
new_df['CNN_F1'] = y_pred_new[:, 1]
new_df['DT_Accuracy'] = y_pred_new[:, 2]
new_df['DT_F1'] = y_pred_new[:, 3]

# 保存结果
new_df.to_csv('predicted_redwine_with_metrics.csv', index=False)

print("Predictions added to the new dataset and saved as 'predicted_redwine_with_metrics.csv'")

# 可视化结果
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Predicted Metrics for Red Wine Dataset', fontsize=16)

# CNN_Accuracy
sns.barplot(x=new_df.index, y='CNN_Accuracy', data=new_df, ax=axes[0, 0])
axes[0, 0].set_title('CNN Accuracy')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Accuracy')

# CNN_F1
sns.barplot(x=new_df.index, y='CNN_F1', data=new_df, ax=axes[0, 1])
axes[0, 1].set_title('CNN F1 Score')
axes[0, 1].set_xlabel('Sample Index')
axes[0, 1].set_ylabel('F1 Score')

# DT_Accuracy
sns.barplot(x=new_df.index, y='DT_Accuracy', data=new_df, ax=axes[1, 0])
axes[1, 0].set_title('DT Accuracy')
axes[1, 0].set_xlabel('Sample Index')
axes[1, 0].set_ylabel('Accuracy')

# DT_F1
sns.barplot(x=new_df.index, y='DT_F1', data=new_df, ax=axes[1, 1])
axes[1, 1].set_title('DT F1 Score')
axes[1, 1].set_xlabel('Sample Index')
axes[1, 1].set_ylabel('F1 Score')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()