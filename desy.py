import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')

# 绘图的特征
features = ['Age', 'Balance', 'CreditScore', 'Tenure', 'EstimatedSalary', 'NumOfProducts', 'Gender']

for feature in features:
    total_counts = data[feature].value_counts().sort_index()
    
    exited_counts = data[data['Exited'] == 1][feature].value_counts().sort_index()#计算Exited为1的特征的频数

    plt.figure(figsize=(10, 6))
    plt.plot(total_counts.index, total_counts.values, label='Total')
    plt.plot(exited_counts.index, exited_counts.values, label='Exited=1')

    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(f'{feature} Distribution')
    plt.legend()

    plt.show()