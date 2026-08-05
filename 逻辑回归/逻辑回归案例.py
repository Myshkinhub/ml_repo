# 导包
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1.加载数据
# 2.数据预处理
# 3.特征工程（特征提取、特征预处理标准化、）
# 4.模型训练
# 5.模型预测
# 6.模型评估


# 1.加载数据
data = pd.read_csv('./data/breast-cancer-wisconsin.csv')
data.info()

# 2.数据预处理
data.replace('?',np.nan,inplace=True) # inplace 是否替换原数据
data.dropna(inplace=True) # 删除有缺失值的行
data.info()

# 3.特征工程
# 3.1 特征提取
x = data.iloc[:,1:-1]
y = data.Class
# print(x)
# print(y)
# 3.2 特征预处理
x_train,x_text,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
# 3.3 标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_text)

# 4.模型训练
estimator = LogisticRegression()
estimator.fit(x_train,y_train)

# 5.模型预测
y_pred = estimator.predict(x_test) # 阈值默认0.5
print(f"预测值为{y_pred}")

# 6.模型评估
print(f"预测前评估,正确率:{estimator.score(x_test,y_test)}")
print(f"预测后评估,正确率:{accuracy_score(y_test,y_pred)}")

# 逻辑回归模型用正确率评估的结果不精准,逻辑回归模型主要用于二分类
#例子：比如 100 个样本中只有 3 个癌症患者，模型哪怕把所有样本都预测为“健康”，准确率也有 97%，但这是一个完全无用的“废物模型”。
# 在医疗数据中，我们更关心召回率（Recall）——即“真正把癌症病人找出来的能力”。
# 通过混淆矩阵来评测，正确率，召回率，F1值，ROC曲线，AUC值