from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def split_train_test():   # 切分训练集和测试集
    # 1.加载数据集
    iris_data = load_iris()

    print(iris_data.feature_names)

    # # 2.数据预处理
    # x_train, x_test, y_train, y_test = train_test_split(iris_data.data,iris_data.target, test_size=0.2,random_state=10)
    # print(f"训练集的特征: {x_train}\n个数: {len(x_train)}")
    # print(f"训练集的标签: {y_train}\n个数: {len(y_train)}")
    # print(f"测试集的特征: {x_test}\n个数:{len(x_test)}")

def iris_evaluate_test():
    # 1.加载数据集
    iris_data = load_iris()
    print(iris_data.DESCR)

    # 2.数据预处理
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,
                                                        random_state=20)

    # 3.特征工程
    # （1）特征提取：该数据源只有4个特征列，均要使用，无需提取
    # （2）特征预处理（标准化）
    # 3.1创建标准化对象
    transfer = StandardScaler()
    # 3.2对特征列进行标准化（x_train,x_test）
    # fit_transform 兼具拟合（fit:只学习参数不改数据）和转换（transform：只改数据不学习）的功能
    x_train = transfer.fit_transform(x_train)
    # 对于测试集只能用transform使用训练集上拟合好的参数来转换
    x_test = transfer.transform(x_test)

    # 4.模型训练
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train) # 传入训练集的特征数据，训练集的标签数据

    # 5.模型预测
    # 场景1：对测试集的30条数据进行预测
    y_pred = estimator.predict(x_test)
    print(f"预测结果为：{y_pred}")

    # 场景2：数据集之外的数据进行预测
    # 5.1自定义数据集
    new_data = [[6.0,1.9,4.3,2.0]]
    # 5.2对自定义数据进行标准化
    new_data = transfer.transform(new_data)
    y_pred_new = estimator.predict(new_data)
    print(f"预测结果为：{y_pred_new}")

    # 5.5查看各分类的预测概率
    y_pred_proba = estimator.predict_proba(new_data)
    print(f"各分类的预测概率为: {y_pred_proba}")

    # 6.模型评估
    # 方式1: 直接进行评分，基于 测试集的特征 和 测试集的标签
    print(f"正确率: {estimator.score(x_test, y_test)}")
    # 方式2: 基于 测试集的标签 和 预测结果 进行评分
    print(f"正确率: {accuracy_score(y_test, y_pred)}")

if __name__ == '__main__':
    iris_evaluate_test()