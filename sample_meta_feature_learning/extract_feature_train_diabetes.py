import pandas as pd
import numpy as np
from pymfe.mfe import MFE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 加载数据集
file_path = 'diabetes_sets.csv'  # 请确保文件路径正确
data = pd.read_csv(file_path)

# 检查数据结构
print(data.head())

# 准备数据（假设最后一列为目标变量）
X = data.iloc[:, :-1].values  # 去掉最后一列
y = data.iloc[:, -1].values  # 最后一列作为目标变量

# 将标签转换为数值类型（如果是分类问题）
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

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
    ft_values = [f"diabetes_{i + 1}"] + list(ft_values)  # 添加数据集标识
    meta_features_list.append(ft_values)

# 创建一个DataFrame来保存所有子集的元特征
columns = ['Dataset'] + ft_names
meta_features_df = pd.DataFrame(meta_features_list, columns=columns)


# 定义1D-CNN模型
class CNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16 * (input_size // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * (input_size // 2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_cnn(X_train, y_train, X_val, y_val, input_size, num_classes):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
                                torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = CNN1D(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, f1


def train_decision_tree(X_train, y_train, X_val, y_val):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    return accuracy, f1


# 训练和评估模型，并添加准确率和F1分数到CSV文件
cnn_accuracies, cnn_f1s, dt_accuracies, dt_f1s = [], [], [], []
for i, (X_subset, y_subset) in enumerate(zip(X_splits, y_splits)):
    X_train, X_val, y_train, y_val = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

    # 训练1D-CNN模型
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    cnn_accuracy, cnn_f1 = train_cnn(X_train, y_train, X_val, y_val, input_size, num_classes)
    cnn_accuracies.append(cnn_accuracy)
    cnn_f1s.append(cnn_f1)

    # 训练决策树模型
    dt_accuracy, dt_f1 = train_decision_tree(X_train, y_train, X_val, y_val)
    dt_accuracies.append(dt_accuracy)
    dt_f1s.append(dt_f1)

    # 打印每个数据集的准确率和F1分数
    print(f"Dataset {i + 1} - CNN Accuracy: {cnn_accuracy}, CNN F1: {cnn_f1}")
    print(f"Dataset {i + 1} - DT Accuracy: {dt_accuracy}, DT F1: {dt_f1}")

    # 保存准确率和F1分数到meta_features_df
    meta_features_df.loc[i, 'CNN_Accuracy'] = cnn_accuracy
    meta_features_df.loc[i, 'CNN_F1'] = cnn_f1
    meta_features_df.loc[i, 'DT_Accuracy'] = dt_accuracy
    meta_features_df.loc[i, 'DT_F1'] = dt_f1

# 保存包含准确率和F1分数的CSV文件
output_file_path = 'extracted_meta_features_diabetes_subsets_with_metrics.csv'
meta_features_df.to_csv(output_file_path, index=False)

print(f"元特征及其准确率和F1分数已保存到 {output_file_path}")

# 可视化准确率和F1分数
datasets = [f'Dataset {i + 1}' for i in range(5)]

plt.figure(figsize=(14, 8))

# 可视化准确率
plt.subplot(2, 1, 1)
width = 0.35
x = np.arange(len(datasets))
plt.bar(x - width / 2, cnn_accuracies, width, label='CNN Accuracy')
plt.bar(x + width / 2, dt_accuracies, width, label='DT Accuracy')
plt.ylabel('Accuracy')
plt.title('Accuracy of CNN and Decision Tree')
plt.xticks(x, datasets)
plt.legend()

# 可视化F1分数
plt.subplot(2, 1, 2)
plt.bar(x - width / 2, cnn_f1s, width, label='CNN F1')
plt.bar(x + width / 2, dt_f1s, width, label='DT F1')
plt.ylabel('F1 Score')
plt.title('F1 Score of CNN and Decision Tree')
plt.xticks(x, datasets)
plt.legend()

plt.tight_layout()
plt.show()