import pickle
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from jinja2.utils import open_if_exists


def get_clean_data():
    # 从指定路径读取CSV格式的原始数据
    data = pd.read_csv("data/data.csv")

    # 删除无用的列 "Unnamed: 32"，这通常是导入时产生的空列
    data = data.drop(["Unnamed: 32"], axis=1)

    # 将目标列 "diagnosis" 的字符标签映射为数值：
    # 'M' (恶性 Malignant) -> 1
    # 'B' (良性 Benign) -> 0
    data["diagnosis"] = data["diagnosis"].map({'M': 1, 'B': 0})

    # 返回清洗后的数据
    return data


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    # 初始化一个空字典，用于存储用户输入的特征值
    input_dict = {}

    # 遍历 slider_labels 列表，其中每个元素是一个 (显示标签, 数据列名) 的元组
    for label, key in slider_labels:
        # 在侧边栏创建一个滑动条，并将用户选择的值存入 input_dict
        input_dict[key] = st.sidebar.slider(
            label=label,  # 滑动条的显示名称（用户看到的文字）
            min_value=float(0),  # 滑动条的最小值，固定为 0
            max_value=float(data[key].max()),  # 滑动条的最大值，取该特征在数据集中的最大值
            value=float(data[key].mean())  # 滑动条的默认值，取该特征在数据集中的平均值
        )

    # 返回包含所有特征及其用户输入值的字典
    return input_dict


def get_scale_value(input_dict):
    # 加载并获取清洗后的原始数据集
    data = get_clean_data()

    # 从数据集中分离出特征列（移除诊断结果列diagnosis）
    x = data.drop(labels=["diagnosis"], axis=1)

    # 初始化一个空字典，用于存储标准化后的特征值
    scaled_dict = {}

    # 遍历用户输入的每个特征及其原始值
    for key, value in input_dict.items():
        # 获取该特征在原始数据集中的最大值和最小值
        max_value = x[key].max()
        min_value = x[key].min()

        # 使用Min-Max标准化公式将原始值缩放到[0, 1]区间
        # 公式：scaled_value = (原始值 - 最小值) / (最大值 - 最小值)
        scaled_value = (value - min_value) / (max_value - min_value)

        # 将标准化后的值存入字典
        scaled_dict[key] = scaled_value

    # 返回包含所有标准化后特征值的字典
    return scaled_dict


def get_radar_chart(input_data):
    # 定义极坐标图的维度分类标签（对应乳腺癌特征的大类）
    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    # 对用户输入的原始数据进行标准化处理，将所有特征值缩放到[0, 1]区间
    # 这样才能在同一个雷达图上进行公平的可视化对比
    input_data = get_scale_value(input_data)

    # 创建一个空的Plotly图形对象，用于绘制极坐标图
    fig = go.Figure()

    # 第一个极坐标轨迹：绘制各特征的“均值(Mean)”数据
    # 反映该特征的 “整体水平”，是判断肿瘤特征的基准数据
    fig.add_trace(go.Scatterpolar(
        r=[  # 极坐标图的径向数值（对应各特征的均值）
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,  # 极坐标图的角度轴标签（对应categories列表）
        fill='toself',  # 填充轨迹内部区域，让图形更直观
        line=dict(color='#1f77b4', width=3),  # 深蓝色，加粗线条
        name='Mean Value'  # 该轨迹的名称（图例显示）
    ))

    # 第二个极坐标轨迹：绘制各特征的“标准误差(SE)”数据
    # 补充均值的信息，反映肿瘤特征的 “稳定性”，是判断肿瘤异质性的重要指标
    fig.add_trace(go.Scatterpolar(
        r=[  # 极坐标图的径向数值（对应各特征的标准误差）
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,  # 角度轴标签与第一个轨迹保持一致
        fill='toself',
        line=dict(color='#2ca02c', width=3),  # 亮绿色，加粗线条
        name='Standard Error'  # 图例显示名称
    ))

    # 第三个极坐标轨迹：绘制各特征的“最差值(Worst)”数据
    # 聚焦肿瘤最异常的部分，是区分良恶性肿瘤的核心指标
    fig.add_trace(go.Scatterpolar(
        r=[  # 极坐标图的径向数值（对应各特征的最差值）
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        line=dict(color='#ff7f0e', width=3),  # 橙色，加粗线条
        name='Worst Value'  # 图例显示名称
    ))

    # 更新图形布局，优化极坐标图的显示效果
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,  # 显示径向坐标轴
                range=[0, 1]  # 设置径向轴的数值范围为0到1（需确保数据已标准化）
            )),
        showlegend=True  # 显示图例，方便区分三个轨迹
    )

    # 返回绘制好的极坐标图对象，用于在Streamlit页面中展示
    return fig

def add_prediction(input_data):
    # 加载模型和标准化器
    # 从指定路径加载预训练好的机器学习模型文件（model.pkl）
    model = pickle.load(open('model/model.pkl', 'rb'))
    # 加载训练时使用的标准化器文件（scaler.pkl），用于统一输入数据的尺度
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))

    # 直接从字典创建 DataFrame，保持列名和顺序
    # 将用户输入的特征字典（input_data）转换为 Pandas DataFrame 格式
    # 注：[input_data] 是为了让字典变成一行数据，符合模型输入要求
    input_df = pd.DataFrame([input_data])

    # 确保列名顺序与训练时一致
    # scaler.feature_names_in_ 存储了训练时特征列的名称和顺序
    # 重新排序输入数据的列，保证和模型训练时的特征顺序完全匹配，避免预测错误
    input_df = input_df[scaler.feature_names_in_]

    # 标准化输入数据
    # 使用训练好的标准化器对用户输入数据进行缩放，统一到模型训练时的尺度
    input_array_scaled = scaler.transform(input_df)

    # 进行预测
    # 调用模型的predict方法，传入标准化后的输入数据，得到预测结果（0=良性，1=恶性）
    prediction = model.predict(input_array_scaled)

    # 在Streamlit页面上添加二级标题，标题内容为“Cell cluster prediction（细胞簇预测）”
    # subheader是比header小一级的标题，用于划分页面内容模块
    st.subheader("Cell cluster prediction")

    # 在页面上显示文本：“The cell cluster is:（细胞簇类型为：）”
    # 作为预测结果的提示文字，引导用户查看后续结论
    st.write("The cell cluster is:")

    # 根据预测结果显示诊断结论
    # prediction[0] 取预测结果数组的第一个（也是唯一）元素（模型输出为二维数组，单样本时只有1行）
    # 0代表良性肿瘤，1代表恶性肿瘤（该映射关系由模型训练时的标签定义）
    if prediction[0] == 0:
        # 若预测值为0，在页面显示“Benign（良性）
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        # 若预测值为1，在页面显示“Malignant（恶性）
        st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)

    # 显示良性肿瘤的概率
    # model.predict_proba() 返回模型对输入样本的类别概率分布，格式为[[良性概率, 恶性概率]]
    # [0][0] 取第一个样本（唯一样本）的第一个类别（良性）的概率值
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])

    # 显示恶性肿瘤的概率
    # [0][1] 取第一个样本的第二个类别（恶性）的概率值
    st.write("Probability of being malignant: ", model.predict_proba(input_array_scaled)[0][1])

    # 在页面显示免责声明文本
    # 明确说明该应用仅作为医疗专业人员的辅助工具，不能替代专业诊断，规避使用风险
    st.write(
        "This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

    # 返回预测结果（数值形式：0/1），供后续可能的逻辑处理（如日志记录、数据存储等）
    return prediction


def main():
    # 配置Streamlit页面的基础设置（网页的全局属性）
    st.set_page_config(
        page_title="Breast Cancer Predictor",  # 设置浏览器标签页的标题为“乳腺癌预测器”
        layout="wide",  # 启用宽屏布局，最大化利用页面空间，适配大屏显示
        initial_sidebar_state="expanded"  # 页面加载时侧边栏默认展开，方便用户直接调整参数
    )

    # 加载并应用自定义CSS样式文件，美化页面UI
    # 打开assets目录下的style.css文件，读取其中的样式代码
    with open("assets/style.css") as file:
        # 通过markdown将CSS样式注入页面，unsafe_allow_html=True允许执行HTML/CSS代码
        st.markdown("<style>{}</style>".format(file.read()), unsafe_allow_html=True)

    # 调用add_sidebar()函数，创建侧边栏并获取用户输入的特征数据
    # 该函数返回的input_data是包含所有特征值的字典
    input_data = add_sidebar()

    # 创建一个容器（container），用于分组展示标题和说明文本，让页面结构更规整
    with st.container():
        st.title("Breast Cancer Predictor")  # 设置页面主标题
        # 显示应用的功能说明文本，告知用户应用的用途和使用方式
        st.write(
            "Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")

    # 将页面主体区域分为两列，列宽比例为4:1（左侧宽，右侧窄）
    # 左侧用于展示雷达图，右侧用于展示预测结果，合理分配视觉空间
    col1, col2 = st.columns([4, 1])

    # 在第一列（宽列）中展示雷达图
    with col1:
        # 调用get_radar_chart()函数，传入用户输入数据，生成特征雷达图
        radar_chart = get_radar_chart(input_data)
        # 在页面上渲染Plotly雷达图
        st.plotly_chart(radar_chart)

    # 在第二列（窄列）中展示预测结果
    with col2:
        # 调用add_prediction()函数，传入用户输入数据，执行预测并展示结果
        add_prediction(input_data)

# 当脚本被直接运行时，调用main函数
if __name__ == '__main__':
    main()