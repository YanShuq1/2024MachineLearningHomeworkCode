import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize

# 读取数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 数据处理
train_df = pd.get_dummies(train_df, columns=['Gender', 'Geography'])
test_df = pd.get_dummies(test_df, columns=['Gender', 'Geography'])

test_customer_ids = test_df['id']

train_df = train_df.drop(['id', 'CustomerId', 'Surname'], axis=1)
test_df = test_df.drop(['id', 'CustomerId', 'Surname'], axis=1)

X = train_df.drop('Exited', axis=1)
y = train_df['Exited']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 划分训练集和验证集
X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# L2正则化和Nesterov Momentum的逻辑回归类
class LogisticRegressionWithSGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization_strength=0.1, momentum_factor=0.9):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization_strength = regularization_strength
        self.momentum_factor = momentum_factor
        self.weights = None
        self.bias = None
        self.weight_momentums = None
        self.bias_momentum = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.weight_momentums = np.zeros(n_features)
        self.bias_momentum = 0

        for _ in range(self.n_iterations):
            # 预测和计算梯度
            linear_model = np.dot(X, self.weights + self.weight_momentums * self.momentum_factor) + self.bias
            y_predicted = self._sigmoid(linear_model)
            dw = (np.dot(X.T, (y_predicted - y)) / n_samples) + self.regularization_strength * self.weights
            db = (np.sum(y_predicted - y) / n_samples) + self.regularization_strength * self.bias

            # 更新动量
            self.weight_momentums = self.momentum_factor * self.weight_momentums - self.learning_rate * dw
            self.bias_momentum = self.momentum_factor * self.bias_momentum - self.learning_rate * db

            # 更新权重和偏置
            self.weights += self.weight_momentums
            self.bias += self.bias_momentum

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

# 定义目标函数
def objective(params):
    learning_rate, n_iterations, regularization_strength, momentum_factor = params

    model = LogisticRegressionWithSGD(learning_rate=learning_rate, n_iterations=n_iterations, regularization_strength=regularization_strength, momentum_factor=momentum_factor)
    model.fit(np.asarray(X_train_scaled), np.asarray(y_train))

    y_val_pred = model.predict(np.asarray(X_val_scaled))
    val_auc = roc_auc_score(y_val, y_val_pred)

    return -val_auc

# 超参数范围
space = [(0.0001, 0.3, 'log-uniform'),
         (100, 1000),
         (1e-6, 5e-1, 'log-uniform'),
         (0.01, 0.9)]

# 贝叶斯优化
res = gp_minimize(objective, space, n_calls=20, random_state=0)

print("Best parameters: {}".format(res.x))

model = LogisticRegressionWithSGD(learning_rate=res.x[0], n_iterations=res.x[1], regularization_strength=res.x[2], momentum_factor=res.x[3])
model.fit(np.asarray(X_train_scaled), np.asarray(y_train))

y_val_pred = model.predict(np.asarray(X_val_scaled))
val_auc = roc_auc_score(y_val, y_val_pred)
print(f'Validation AUC: {val_auc}')

# 生成提交文件
test_df_scaled = scaler.transform(test_df)
test_predictions = model.predict(np.asarray(test_df_scaled))

submission_df = pd.DataFrame({'id': test_customer_ids, 'Exited': test_predictions})
submission_df.to_csv('submission.csv', index=False)
