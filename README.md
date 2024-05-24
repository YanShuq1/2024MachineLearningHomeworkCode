# 2024机器学习大作业

## 选题：[银行用户流失预测 Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/competitions/playground-series-s4e1)

## 文件结构说明：
- `train.csv:`这个文件包含了训练集数据，每一行代表一个客户信息，包括客户的各种特征以及是否已经流失（Exited）。
- `test.csv:`这个文件包含了测试集数据，也是每一行代表一个客户信息，但没有包含是否已经流失（Exited）的信息。
- `submission.csv:`这个文件是测试结果文件，包含了对测试集中每个客户的流失概率的预测值。
- `desy.py`: 分析训练集中各个特征的分布情况，并将流失客户（Exited=1）与总客户数进行对比，以便于理解不同特征对客户流失的影响，并绘制相关图表。
- `form_o.py:`实现了一个二元分类器（LogisticRegressionWithSGD），使用了随机梯度下降（SGD）优化算法结合L2正则化和Nesterov Momentum。
- `form.py:`实现了一个类似于前一个文件的功能，但是与常规处理不同的是没有使用贝叶斯优化来选择超参数，而是直接在代码中指定了超参数的值。
- `mlp.py:`实现了一个多层感知机（MLP），用于二元分类任务。
- `svm.py:`实现了一个简单的支持向量机（SVM）二元分类器，并将其应用于银行客户流失预测任务中。
- `xgb.py:`实现了使用XGBoost建立分类模型，进行交叉验证和预测的流程。
- `Figure_by_desy:`此文件夹存储`desy.py`生成的折线图


## 补充：
本人使用的开发环境为mac+vscode+py@3.9.13，有一些库如cupy，keras等这些库不支持，若你的开发环境支持这些库的话还可以进一步优化代码处理速度

# 2024MachineLearningHomeworkCode
