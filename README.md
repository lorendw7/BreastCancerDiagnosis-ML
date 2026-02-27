# BreastCancerDiagnosis-ML
基于机器学习的乳腺癌诊断预测工具

## 项目概述
本项目是一款由机器学习驱动的乳腺癌辅助诊断工具，旨在为医疗从业者提供乳腺肿块良恶性的预测参考：
- 核心功能：输入乳腺肿块的各项测量指标，模型可预测肿块为良性（benign）或恶性（malignant），并输出对应的概率值
- 可视化能力：通过雷达图直观展示输入数据的特征分布
- 使用方式：支持手动输入测量数据，也可预留对接细胞学实验室设备的扩展接口（注：实验室设备对接功能非本应用范畴）

### 重要说明
本项目仅为机器学习领域的教育实践项目，基于公开的「威斯康星乳腺癌诊断数据集」（Breast Cancer Wisconsin (Diagnostic) Data Set）开发。该数据集不具备专业医疗级可靠性，**严禁将本工具用于临床诊断等专业医疗场景**。

## 环境安装
推荐使用 `conda` 创建虚拟环境管理依赖，步骤如下：

### 1. 创建虚拟环境
创建名为 `breast-cancer-diagnosis` 的虚拟环境，指定 Python 版本为 3.10：
```bash
conda create -n breast-cancer-diagnosis python=3.10
```

### 2. 激活虚拟环境
```bash
conda activate breast-cancer-diagnosis
```

### 3.安装依赖包
```bash
pip install -r requirements.txt
```

### 3.使用方法
```bash
streamlit run app/main.py
```
