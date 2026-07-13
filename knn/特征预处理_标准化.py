
from sklearn.preprocessing import StandardScaler

# 准备数据集（标准化前的原始数据）
x_train = [[90,2,10,40],[60,4,15,45],[75,3,13,46]]

# 创建标准化对象
transfer = StandardScaler()

# 对原数据集进行标准化操作
x_train_new = transfer.fit_transform(x_train)

print(x_train_new)

print(f"数据集的均值：{transfer.mean_.round(2)}")
print(f"数据集的方差：{transfer.var_.round(2)}")