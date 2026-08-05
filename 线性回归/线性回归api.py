from sklearn.linear_model import LinearRegression

# 1.加载数据
x_train = [[160],[166],[172],[174],[180]] # 训练集的特征
y_train = [56.3,60.6,65.1,68.5,75]     # 训练集的标签
x_test = [[176]]                          # 测试集的特征


# 2.数据预处理
# 3.特征工程（特征提取，特征预处理）


# 4.模型训练
# 4.1创建模型对象
estimator = LinearRegression()
# 4.2具体训练模型
estimator.fit(x_train, y_train)

# 5.模型预测
print(f"权重: {estimator.coef_}")
print(f"偏置值: {estimator.intercept_}")
y_pred = estimator.predict(x_test)

print(f"预测结果是: {y_pred}")
