import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb

# 读取数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 数据处理
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

X = combined_df.loc['train']
X_test = combined_df.loc['test']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# XGBoost 模型
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc'
)

# 交叉验证
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc')
print("Cross-Validation AUC scores:", cv_scores)
print("Mean Cross-Validation AUC score:", np.mean(cv_scores))

# 训练模型
xgb_model.fit(X_train, y_train)

# 验证集上的预测
y_val_pred = xgb_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
print("Validation AUC:", val_auc)

# 测试集上的预测
test_predictions = xgb_model.predict_proba(X_test_scaled)[:, 1]


submission_df = pd.DataFrame({'id': test_customer_ids, 'Exited': test_predictions})
submission_df.to_csv('submission.csv', index=False)
