import matplotlib.pyplot as plt

# 存储模型性能数据
models = [
    "ExtraTreesMSE BAG L1", "RandomForestMSE BAG L1", "NeuralNetTorch BAG L1",
    "XGBoost BAG L1", "LightGBM BAG L1", "KNeighborsUnif BAG L1",
    "LightGBMLarge BAG L1", "KNeighborsDist BAG L1", "CatBoost BAG L1",
    "NeuralNetFastAI BAG L1", "LightGBMXT BAG L1", "NeuralNetTorch BAG L2",
    "NeuralNetFastAI BAG L2", "WeightedEnsemble L2", "RandomForestMSE BAG L2",
    "LightGBMLarge BAG L2", "ExtraTreesMSE BAG L2", "XGBoost BAG L2",
    "LightGBMXT BAG L2", "CatBoost BAG L2", "LightGBM BAG L2", "WeightedEnsemble L3"
]

rmse = [
    26.29, 26.28, 25.64, 25.13, 24.74, 24.68,
24.66, 24.50, 24.36, 24.22, 24.19, 23.76,
22.94, 22.60, 22.58, 22.50, 22.50, 22.49,
22.42, 22.41, 22.40, 22.30
]

# 为WeightedEnsemble L3设置特殊颜色和标签
colors = ['#1f77b4' if model != "WeightedEnsemble L3" else '#d62728' for model in models]
labels = ['' if model != "WeightedEnsemble L3" else model for model in models]

# 创建柱状图
fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
bars = ax.bar(models, rmse, color=colors)

# 设置X轴标签倾斜
plt.xticks(rotation=20, ha="right")
# 调整图形底部边距
plt.subplots_adjust(bottom=0.25)
# # 在柱子顶部添加数值标签
# for i, bar in enumerate(bars):
#     ax.text(
#     bar.get_x() + bar.get_width() / 2,
#     bar.get_height(),
#     f"{rmse[i]}",
#     ha="center",
#     va="bottom",
#     fontsize=9,)

# 添加标题和坐标轴标签
# plt.title("Comparison of Model Performance")
plt.xlabel("Model")
plt.ylabel("RMSE")


# 保存图像到指定路径
plt.savefig('/home/dell/wangzhen/RealQED(2.17)/test/result/autogluon_model.png')
# 显示图
plt.show()
