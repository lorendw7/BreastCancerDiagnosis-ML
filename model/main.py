import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_model(data):
    # 从数据中分离特征（X）和目标变量（y）
    # 特征X：除了"diagnosis"列之外的所有列
    x = data.drop("diagnosis", axis=1)
    # 目标y："diagnosis"列，代表诊断结果（M=恶性，B=良性）
    y = data["diagnosis"]

    # 对特征进行标准化处理（均值为0，方差为1）
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # 将数据划分为训练集（80%）和测试集（20%）
    # random_state=42 确保划分结果可复现
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # 初始化逻辑回归模型
    model = LogisticRegression()
    # 使用训练集数据训练模型
    model.fit(x_train, y_train)

    # 在测试集上评估模型
    y_pred = model.predict(x_test)
    print("Accuracy of the model: ", accuracy_score(y_test, y_pred))
    print("Classification Report: \n", classification_report(y_test, y_pred))

    # 返回训练好的模型
    return model, scaler


def get_clean_data():
    # 从指定路径读取CSV格式的原始数据
    data = pd.read_csv("../data/data.csv")

    # 删除无用的列 "Unnamed: 32"，这通常是导入时产生的空列
    data = data.drop(["Unnamed: 32"], axis=1)

    # 将目标列 "diagnosis" 的字符标签映射为数值：
    # 'M' (恶性 Malignant) -> 1
    # 'B' (良性 Benign) -> 0
    data["diagnosis"] = data["diagnosis"].map({'M': 1, 'B': 0})

    # 返回清洗后的数据
    return data

def main():
    # 1. 获取清洗后的数据
    data = get_clean_data()

    # 2. 基于清洗后的数据创建模型和数据标准化器
    model, scaler = create_model(data)

    # 3. 保存训练好的模型到文件
    # 以二进制写入模式打开文件路径 "../model/model.pkl"
    with open("../model/model.pkl", "wb") as file:
        # 使用pickle将模型对象序列化并写入文件
        pickle.dump(model, file)

    # 4. 保存数据标准化器到文件
    # 以二进制写入模式打开文件路径 "../model/scaler.pkl"
    with open("../model/scaler.pkl", "wb") as file:
        # 使用pickle将标准化器对象序列化并写入文件
        pickle.dump(scaler, file)

if __name__ == '__main__':
    main()