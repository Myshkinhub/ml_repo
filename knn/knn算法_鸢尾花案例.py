from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def split_train_test():   # 切分训练集和测试集
    iris_data = load_iris()


    x_train, x_test, y_train, y_test = train_test_split(iris_data.data,iris_data.target, test_size=0.2,random_state=10)
    print(f"训练集的特征: {x_train}\n个数: {len(x_train)}")
    print(f"训练集的标签: {y_train}\n个数: {len(y_train)}")
    print(f"测试集的特征: {x_test}\n个数:{len(x_test)}")

def iris_evaluate_test():
    iris_data = load_iris()


if __name__ == '__main__':
    iris_evaluate_test()