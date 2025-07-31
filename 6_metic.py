import matplotlib.pyplot as plt

# 文件路径定义
predicted_file = "/home/mozhengao/code/particle/data3/script/pic_binary.txt"
ground_truth_file = "/home/mozhengao/code/particle/data3/script/bad.txt"
output = "/home/mozhengao/code/particle/data3/script/accuracy_recall_plot.png"
# 获取 predicted_file 中的行数作为 num
with open(predicted_file, "r") as file:
    predicted_data = [line.strip().split()[0] for line in file if len(line.strip().split()) >= 2]

num = len(predicted_data)

# 获取 ground_truth_file 中的行数
with open(ground_truth_file, "r") as file:
    ground_truth_set = set(line.strip() for line in file)

# 初始化准确率和召回率列表
accuracies = []
recalls = []

# 计算不同预测颗粒数量的准确率和召回率
for i in range(1, num + 1):
    # 取前 i 个形变量大的颗粒作为预测形变颗粒
    predicted_top_i = set(predicted_data[:i])

    # 计算准确率和召回率
    true_positives = len(predicted_top_i & ground_truth_set)  # 模型预测正确的形变颗粒数
    total_predicted = len(predicted_top_i)                   # 模型预测的形变颗粒总数
    total_actual = len(ground_truth_set)                       # 实际形变颗粒的总数

    accuracy = true_positives / total_predicted if total_predicted > 0 else 0
    recall = true_positives / total_actual if total_actual > 0 else 0

    accuracies.append(accuracy)
    recalls.append(recall)

    # 如果 i 等于 ground_truth_file 中的行数，打印准确率、召回率和 F1 指数
    if i == len(ground_truth_set):
        f1_score = (2 * accuracy * recall) / (accuracy + recall) if (accuracy + recall) > 0 else 0
        print(f"模型准确率（Accuracy）：{accuracy:.2%} ({true_positives}/{total_predicted})")
        print(f"模型召回率（Recall）：{recall:.2%} ({true_positives}/{total_actual})")
        print(f"模型 F1 指数（F1 Score）：{f1_score:.2f}")

# 绘制准确率和召回率的图像
plt.figure(figsize=(8, 6))
plt.plot(range(1, num + 1), accuracies, label="Accuracy", color='blue', marker='o',markersize=0.1)
plt.plot(range(1, num + 1), recalls, label="Recall", color='green', marker='s',markersize=0.1)

# 设置图表标题和标签
plt.xlabel("Number of Predicted Particles")
plt.ylabel("Score")
plt.legend()

# 保存图像
plt.tight_layout()
plt.savefig(output)
plt.show()
