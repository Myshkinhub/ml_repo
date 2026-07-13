"""
特征工程的目的和步骤
    目的：利用专业的背景知识 和 技巧处理数据，用于提升模型的性能
    步骤：
        1.特征提取
        2.特征预处理
        3.特征降维
        4.特征选择
        5.特征组合
"""
# 导包
from sklearn.preprocessing import MinMaxScaler

# 准备数据集（归一化之前的原始数据）
x_train = [[90,2,10,40],[60,4,15,45],[75,3,13,46]]

# 创建归一化对象
transfer = MinMaxScaler(feature_range=(0,1)) # 默认[0，1]可以省略

# 对原数据集进行归一化操作
x_train_new = transfer.fit_transform(x_train)

print("处理后的数据集：\n")
print(x_train_new.round(2))


