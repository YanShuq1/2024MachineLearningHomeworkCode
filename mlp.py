import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 读取数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

test_customer_ids = test_df['id']
train_df = train_df.drop(['id', 'CustomerId', 'Surname'], axis=1)
test_df = test_df.drop(['id', 'CustomerId', 'Surname'], axis=1)

X = train_df.drop('Exited', axis=1)
y = train_df['Exited']

combined_df = pd.concat([X, test_df], keys=['train', 'test'])

encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(combined_df[['Geography', 'Gender']]).toarray()
encoded_feature_names = encoder.get_feature_names_out(['Geography', 'Gender'])

encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=combined_df.index)
combined_df = combined_df.drop(['Geography', 'Gender'], axis=1)
combined_df = pd.concat([combined_df, encoded_df], axis=1)

X = combined_df.loc['train'].values
X_test = combined_df.loc['test'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


class MLP:

    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(int)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def score(self, X, y):
        y_pred = self.predict_proba(X)
        return roc_auc_score(y, y_pred)

    def forward(self, X):
        activations = [X]
        inputs = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], W) + b
            inputs.append(z)
            if i == len(self.weights) - 1:
                a = self.sigmoid(z)
            else:
                a = self.relu(z)
            activations.append(a)

        return activations, inputs

    def backward(self, activations, inputs, y):
        m = y.shape[0]
        deltas = [activations[-1] - y]

        for i in range(len(self.weights) - 1, 0, -1):
            if i == len(self.weights) - 1:
                delta = np.dot(deltas[0], self.weights[i].T) * self.sigmoid_derivative(activations[i])
            else:
                delta = np.dot(deltas[0], self.weights[i].T) * self.relu_derivative(activations[i])
            deltas.insert(0, delta)

        dW = []
        db = []

        for i in range(len(self.weights)):
            dW.append(np.dot(activations[i].T, deltas[i]) / m)
            db.append(np.sum(deltas[i], axis=0, keepdims=True) / m)

        return dW, db

    def update_params(self, dW, db, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]

    def fit(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            activations, inputs = self.forward(X)
            dW, db = self.backward(activations, inputs, y)
            self.update_params(dW, db, learning_rate)

            if epoch % 100 == 0:
                predictions = self.predict_proba(X)
                loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict_proba(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)


y_train = y_train.values.reshape(-1, 1)
y_val = y_val.values.reshape(-1, 1)

X_train_np = np.asarray(X_train)
X_val_np = np.asarray(X_val)

input_size = X_train_np.shape[1]
hidden_layers = [500, 200]
output_size = 1
mlp = MLP(input_size, hidden_layers, output_size)

mlp.fit(X_train_np, y_train, epochs=2000, learning_rate=0.01)

y_val_pred = mlp.predict_proba(X_val_np)
val_auc = roc_auc_score(y_val, y_val_pred)
print("Validation AUC:", val_auc)

test_predictions = mlp.predict_proba(X_test_scaled)

submission_df = pd.DataFrame({'id': test_customer_ids, 'Exited': test_predictions.ravel()})
submission_df.to_csv('submission.csv', index=False)
