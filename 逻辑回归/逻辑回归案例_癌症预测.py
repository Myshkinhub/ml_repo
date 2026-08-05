"""
混淆矩阵：
    用来描述真实值与预测值之间的关系
    Positive：正例
    Negative：反例

    1.模拟一般用 分类少的 充当正例
    2.精准率（Precision） = 真正例 在预测正例中的占比 tp / (tp + fp)
    3.召回率（Recall） = 真正例 在 正例中的占比  tp / (tp + fn)
    4.F1值 精准率和召回率的精准平均分 F1 = 2 * Precision * Recall /（Precision + Recall）

"""

from sklearn.metrics import confusion_matrix,precision_score ,recall_score,f1_score
import pandas as pd

y_train = ["恶性","恶性","恶性","恶性","恶性","恶性","良性","良性","良性","良性",]

# 模型A的预测结果
y_pre_A = ["恶性","良性","良性","恶性","恶性","恶性","良性","恶性","良性","良性",]

# 模型B的预测结果
y_pre_B = ["恶性","良性","恶性","良性","良性","恶性","良性","恶性","恶性","良性",]

df_label = ["预测恶性","预测良性"]
df_index = ["恶性","良性"]

cm_A = confusion_matrix(y_train, y_pre_A)
cm_B = confusion_matrix(y_train, y_pre_B)
print(confusion_matrix(y_train, y_pre_A))

# 混淆矩阵转为 DataFrame
df_A = pd.DataFrame(cm_A,index=df_index,columns=df_label)
print(df_A)

# 模型A的精准率、召回率、F1值
print(f"A模型的精准率：{precision_score(y_train, y_pre_A,pos_label="恶性")}")
print(f"A模型的召回率：{recall_score(y_train, y_pre_A,pos_label="恶性")}")
print(f"A模型的F1值：{f1_score(y_train, y_pre_A,pos_label="恶性")}")
