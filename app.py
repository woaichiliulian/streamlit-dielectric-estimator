# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 16:09:54 2025

@author: user
"""


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体（黑体）
matplotlib.rcParams['axes.unicode_minus'] = False    # 正确显示负号


st.title("基于表面反射法的路面介电常数计算器")
st.markdown("""
本小程序用于通过**探地雷达（GPR）信号**估算**路面材料的介电常数**，以辅助分析路面的**密实度**或**含水率**。介电常数作为材料的重要电磁特性表征参数，可用于进一步分析路面密实度或含水率。

---  
🔧 **功能说明：**  
- 通过上传或选择内置的 **铜板反射信号** 与 **路面反射信号**，提取两者反射振幅，基于路表反射法计算介电常数。
- 支持**多列信号**并自动批量处理。
- 支持导出计算结果为 CSV 文件。

📤**方法说明：**

本小程序通过 GPR 信号中铜板与路面反射的最大幅值之比，利用如下公式计算路面介电常数：

$$ \\varepsilon = \\left( \\frac{1 + A_0/A_p}{1 - A_0/A_p} \\right)^2 $$


其中：
- $A_p$ 为铜板反射信号中延时后的最大值；
- $A_0$ 为待测路面信号中延时后的最大值；
- $\\varepsilon$ 越大，表示水分或密度越高。


📌 **使用步骤：**  
1. 选择数据输入方式（上传CSV文件或使用内置示例信号）  
2. 若上传数据，请提供雷达的采样频率 `fs` 与雷达离地高度 `h`；若使用内置信号，程序会自动设定这些参数  
3. 查看程序自动绘制的信号图与最大反射幅值  
4. 查看并下载计算得到的路面**介电常数估算结果**

📤 **数据格式要求：**  
- CSV 文件应为 **每列为一组独立信号**，不带表头。
- 信号单位为**幅值**（如电压、振幅等），无需归一化处理。

---
""")

# ------------------- 1. 输入方式选择 -------------------
input_mode = st.radio(
    "请选择数据输入方式：",
    ("上传CSV文件", "使用内置示例信号")
)



# ------------------- 2. 数据获取 -------------------
if input_mode == "上传CSV文件":
    st.write("请分别上传两份CSV文件：")
    ref_file = st.file_uploader("1️⃣ 上传铜板反射信号数据", type="csv", key="ref")
    test_file = st.file_uploader("2️⃣ 上传待测信号数据（路表反射）", type="csv", key="test")

    if ref_file and test_file:
        # ✅ 原样读取，不做降维处理
        copper = np.loadtxt(ref_file, delimiter=",")
        signal = np.loadtxt(test_file, delimiter=",")
    else:
        st.warning("请同时上传两份数据文件")
        st.stop()


elif input_mode == "使用内置示例信号":
    example_choice = st.selectbox(
        "请选择一个待测信号类型：",
        ("干燥路面", "湿润路面", "压实前路面", "压实后路面")
    )

    # ✅ 原样读取铜板参考信号
    copper = np.loadtxt("copper_ref.csv", delimiter=",")

    # ✅ 原样读取对应路面信号
    if example_choice == "干燥路面":
        signal = np.loadtxt("dry_surface.csv", delimiter=",")
        label = "干燥路面"
    elif example_choice == "湿润路面":
        signal = np.loadtxt("wet_surface.csv", delimiter=",")
        label = "湿润路面"
    elif example_choice == "压实前路面":
        signal = np.loadtxt("before_compaction.csv", delimiter=",")
        label = "压实前路面"
    elif example_choice == "压实后路面":
        signal = np.loadtxt("after_compaction.csv", delimiter=",")
        label = "压实后路面"



if input_mode == "使用内置示例信号":
    fs = 1061 / 5e-9  # sample/s
    h = 0.25  # m
    st.markdown(f"**内置参数已自动设定：**  \n- 采样频率 fs = {fs:.2e} sample/s  \n- 雷达离地高度 h = {h:.2f} m")

elif input_mode == "上传CSV文件":
    st.markdown("请输入下列参数：")

    fs = st.number_input("📏 雷达采样频率 fs（单位：sample/s，例如 2.048e11）")
    h = st.number_input("📐 雷达离地高度 h（单位：m，例如 0.23）")

v = 3e8
t_lim = 2*h/v
i_lim = int(t_lim * fs) 


# ------------------- 3. 数据处理 -------------------

# 判断 copper 是单列还是多列
if copper.ndim == 1:
    # ✅ 单列数据处理
    mean_val = np.mean(copper)
    copper_new = copper - mean_val

else:
    # ✅ 多列数据处理
    # 对每一列做去均值处理
    copper_centered = copper - np.mean(copper, axis=0)  # shape unchanged
    # 每行取平均 → 得到单列均值信号
    copper_new = np.mean(copper_centered, axis=1)

# ✅ 防止 i_lim 超出信号长度，安全计算 Ap
if i_lim >= len(copper_new):
    st.error(f"❌ 错误：计算得到的 i_lim = {i_lim} 已超出铜板信号长度（{len(copper_new)}），请检查采样频率 fs 或信号文件是否正确。")
    st.stop()



# ✅ Ap 取 i_lim 之后的最大绝对值
Ap = np.max(np.abs(copper_new[i_lim:]))

# （可选）显示 i_lim 和 Ap 值
st.markdown(f"已根据 h 和 fs 计算 i_lim = {i_lim}，并从此位置起取最大值 Ap = {Ap:.3f}")

# ✅ 显示处理结果（可选调试输出）
st.subheader("处理后的铜板信号（copper_new）")

fig_ap = go.Figure()

# 添加 copper_new 曲线
fig_ap.add_trace(go.Scatter(
    y=copper_new,
    mode='lines',
    name='去均值后铜板信号',
    line=dict(color='blue')
))

# 添加 Ap 横线
fig_ap.add_trace(go.Scatter(
    x=[0, len(copper_new)],
    y=[Ap, Ap],
    mode='lines',
    name=f'Ap = {Ap:.2f}',
    line=dict(color='red', dash='dash')
))

# 设置图像布局
fig_ap.update_layout(
    title="铜板信号处理结果",
    xaxis_title="样本点",
    yaxis_title="幅值",
    legend_title="图例",
    height=400
)

st.plotly_chart(fig_ap, use_container_width=True)

st.info(f"铜板信号处理完成，最大幅值 Ap = {Ap:.2f}")




if signal.ndim == 1:
    # 单列信号
    A0 = np.max(np.abs(signal[i_lim:]))
else:
    # 多列信号：每列计算 i_lim 后的最大值
    A0 = np.max(np.abs(signal[i_lim:, :]), axis=0)  # A0 是一个数组

# 显示 A0 值（可视化或调试）
st.subheader("A0 计算结果")
if isinstance(A0, np.ndarray):
    for idx, a0_val in enumerate(A0):
        st.markdown(f"**第 {idx + 1} 列信号的 A0：** {a0_val:.3f}")
else:
    st.markdown(f"**单列信号的 A0：** {A0:.3f}")


# ------------------- 计算比值并估算介电常数 -------------------

# 若 A0 是单值，统一转为数组方便处理
if np.isscalar(A0):
    ratio = A0 / Ap
    epsilonac = ((1 + ratio) / (1 - ratio)) ** 2
    st.subheader("介电常数估算结果")
    st.markdown(f"""
    $\\text{{ratio}} = \\frac{{A_0}}{{A_p}} = \\frac{{{A0:.3f}}}{{{Ap:.3f}}} = {ratio:.3f}$

    $\\varepsilon_{{\\mathrm{{ac}}}} = \\left(\\frac{{1 + r}}{{1 - r}}\\right)^2 = {epsilonac:.3f}$
    """)  # ✅ 此行为你缺失的右括号
else:
    # 多列信号：A0 是数组
    ratio = A0 / Ap
    epsilonac = np.power((1 + ratio) / (1 - ratio), 2)

    st.subheader("介电常数估算结果（多列信号）")
    for i in range(len(epsilonac)):
        st.markdown(f"""
        **第 {i + 1} 列:**  
        $\\text{{ratio}} = \\frac{{A_0}}{{A_p}} = \\frac{{{A0[i]:.3f}}}{{{Ap:.3f}}} = {ratio[i]:.3f}$  
        $\\varepsilon_{{\\text{{ac}}}} = \\left(\\frac{{1 + r}}{{1 - r}}\\right)^2 = {epsilonac[i]:.3f}$
        """)

# ------------------- 4.结果可视化 -------------------

st.subheader("信号对比图（铜板 vs 路面）")

fig2 = go.Figure()

# 添加铜板信号（浅蓝色虚线）
fig2.add_trace(go.Scatter(
    y=copper_new,
    mode="lines",
    name="铜板信号",
    line=dict(dash='dash', width=2, color='lightskyblue')  # ✅ 浅蓝色虚线
))

# 添加路面信号
if signal.ndim == 1:
    fig2.add_trace(go.Scatter(
        y=signal,
        mode="lines",
        name="待测信号",
        line=dict(dash='solid', width=2, color='orange'),  # 实线 + 橙色
        opacity=0.8
    ))
else:
    colors = ['orange', 'green', 'red', 'purple', 'brown', 'gray']  # 可扩展
    for i in range(signal.shape[1]):
        fig2.add_trace(go.Scatter(
            y=signal[:, i],
            mode="lines",
            name=f"待测信号 第{i+1}列",
            line=dict(dash='solid', width=2, color=colors[i % len(colors)]),
            opacity=0.7
        ))

fig2.update_layout(
    xaxis_title="采样点",
    yaxis_title="幅值",
    legend_title="信号类型",
    height=400,
    margin=dict(l=30, r=30, t=30, b=30)
)

st.plotly_chart(fig2, use_container_width=True)




# 整理输出表格（仅多列时有用）
if not np.isscalar(epsilonac):
    df = pd.DataFrame({
        'A0': A0,
        'ratio': ratio,
        'epsilon_ac': epsilonac
    })
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 下载介电常数结果表格", csv, "epsilon_results.csv", "text/csv")




st.markdown("✅ **计算完成，可根据需要下载结果或调整输入重新估算。**")




