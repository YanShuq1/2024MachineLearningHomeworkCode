import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

# 读取数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 数据处理
test_customer_ids = test_df['id']
train_df = pd.get_dummies(train_df, columns=['Gender', 'Geography'])
test_df = pd.get_dummies(test_df, columns=['Gender', 'Geography'])

train_df = train_df.drop(['id', 'CustomerId', 'Surname'], axis=1)
test_df = test_df.drop(['id', 'CustomerId', 'Surname'], axis=1)

X = train_df.drop('Exited', axis=1).values.astype(np.float32)
y = train_df['Exited'].values.astype(np.float32)
X_test = test_df.values.astype(np.float32)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def feature_scaling(X):
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X_scaled

X_train_scaled = feature_scaling(X_train)
X_val_scaled = feature_scaling(X_val)
X_test_scaled = feature_scaling(X_test)

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
        self.platt_coef = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_epochs):
            idx = np.random.randint(0, n_samples)  # 随机选择一个样本
            x_i = X[idx]
            condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
            if condition:
                self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
            else:
                self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y[idx]))
                self.bias -= self.learning_rate * y[idx]

        # Platt缩放
        decision_values = np.dot(X, self.weights) - self.bias
        self.platt_coef = self._sigmoid_fit(decision_values, y)

    def _sigmoid_fit(self, decision_values, y):
        lr = LogisticRegression()
        lr.fit(decision_values.reshape(-1, 1), y)
        return lr.coef_[0], lr.intercept_[0]

    def predict_proba(self, X):
        decision_values = np.dot(X, self.weights) - self.bias
        return expit(self.platt_coef[0] * decision_values + self.platt_coef[1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba > 0.5, 1, 0)

# 训练和预测
svm = SVM(learning_rate=0.01, lambda_param=0.01, num_epochs=1000)
svm.fit(X_train_scaled, y_train)

# 验证集预测
y_val_pred_proba = svm.predict_proba(X_val_scaled)

# 计算AUC
val_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f'Validation AUC: {val_auc}')

# 对测试数据进行预测
y_test_pred = svm.predict(X_test_scaled)

# 保存结果
submission_df = pd.DataFrame({'id': test_customer_ids, 'Exited': y_test_pred.ravel()})
submission_df.to_csv('submission.csv', index=False)
