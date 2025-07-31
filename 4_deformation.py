#直接计算型变量不用迭代
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
# 参数
num_point = 8000
template_path = r"/home/mozhengao/code/particle/data2_simulate/10345/j97_binary.png"
particles_path = r"/home/mozhengao/code/particle/data2_simulate/pic_part/Bottom_Right"
output_file_path = r"/home/mozhengao/code/particle/data2_simulate/pic_part/TBottom_Right_sort.txt"  # 保存的txt文件路径
outputfile_region = r"/home/mozhengao/code/particle/data2_simulate/pic_part/Bottom_Right_region.txt" #保存形变情况的
output_all = r"/home/mozhengao/code/particle/data2_simulate/pic_part/Bottom_Right.png"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

deformation_accumulator = np.zeros(num_point, dtype=np.float32)
count_accumulator = np.zeros(num_point, dtype=np.float32)

image = Image.open(template_path)
img_width, img_height = image.size

def update_template_deformation(file_name, B_points, template_points, region_labels, deformation_accumulator, count_accumulator):
    """
    计算当前图片对应的模板颗粒形变值，并更新累积形变值（计算平均形变）

    参数:
    - file_name: str, 当前处理的图片文件名
    - B_points: ndarray, 当前图片的点云数据 (N, 3)
    - template_points: ndarray, 模板点云数据 (N, 3)
    - region_labels: ndarray, 指示每个点属于哪个区域 (N,)
    - deformation_accumulator: ndarray, 存储所有图片累计形变量 (N,)
    - count_accumulator: ndarray, 存储每个点的累计计算次数 (N,)

    返回:
    - updated_deformation: ndarray, 更新后的模板平均形变量 (N,)
    """
    # 确保 B_points 和 template_points 是 NumPy 数组
    if isinstance(B_points, torch.Tensor):
        B_points = B_points.detach().cpu().numpy()
    if isinstance(template_points, torch.Tensor):
        template_points = template_points.detach().cpu().numpy()

    B_points = B_points.squeeze(0)  # 变成 (8000, 2)
    template_points = template_points.squeeze(0)  # 变成 (8000, 2)

    deformation_current = np.linalg.norm(B_points - template_points, axis=1)  # 确保输出形状是 (N,)

    # 累加形变量
    deformation_accumulator += deformation_current
    count_accumulator += 1  # 记录每个点被处理的次数

    # 计算当前的平均形变值
    updated_deformation = deformation_accumulator / count_accumulator

    return updated_deformation


def plot_deformation_with_template(points, deformations, output_all):
    """
    绘制模板点云的形变量可视化

    参数:
    - points: Tensor/ndarray, 模板点云数据 (N, 3)
    - deformations: Tensor/ndarray, 每个点的形变量 (N,)
    - output_path: str, 保存可视化图像的路径
    """
    plt.figure(figsize=(8, 8))

    # 🚀 确保 points 和 deformations 在 CPU 并转为 NumPy 数组
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if isinstance(deformations, torch.Tensor):
        deformations = deformations.detach().cpu().numpy()

    # 确保 points 形状是 (N, 2) 而不是 (1, N, 2)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points.squeeze(0)

    # 确保 deformations 形状是 (N, 1) 或 (N,)
    if deformations.ndim == 2 and deformations.shape[1] == 1:
        deformations = deformations.squeeze(1)

    # 确保形状匹配
    assert deformations.shape[0] == points.shape[0], f"Shape mismatch: {deformations.shape} vs {points.shape}"

    # 归一化形变量，调整点大小和透明度
    point_sizes = (deformations - deformations.min()) / (deformations.max() - deformations.min()) * 20 + 0.1
    point_alphas = (deformations - deformations.min()) / (deformations.max() - deformations.min()) * 0.8 + 0.2

    # 绘制点云
    scatter = plt.scatter(
        points[:, 1],  # 交换 x 和 y
        -points[:, 0],  # 负号翻转 y 轴
        c=deformations, 
        cmap="viridis", 
        s=point_sizes, 
        alpha=point_alphas, 
        edgecolors="face",
    )

    plt.axis("equal")
    plt.axis("off")

    # 保存图像
    plt.savefig(output_all, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

# 区域划分函数
def divide_points_into_quadrants(points, img_width, img_height):
    region_labels = []
    for point in points:
        x, y = point
        if x < img_width / 2 and y < img_height / 2:
            region_labels.append(0)  # 左上
        elif x < img_width / 2 and y >= img_height / 2:
            region_labels.append(1)  # 右上
        elif x >= img_width / 2 and y < img_height / 2:
            region_labels.append(2)  # 左下
        else:
            region_labels.append(3)  # 右下
    return np.array(region_labels)

# 区域形变计算函数
def calculate_region_deformation(file_name, current_points, original_points, region_labels, outputfile_region, scale_max=100, b=3.5):
    """
    计算四个区域的平均形变值，放大较大区域的数值，并计算占比，然后写入文件（覆盖相同文件名的数据）。

    参数:
    - file_name: str, 当前处理的文件名
    - current_points: ndarray, 变形后的点云数据 (N, 3)
    - original_points: ndarray, 原始点云数据 (N, 3)
    - region_labels: ndarray, 指示每个点属于哪个区域 (N,)
    - scale_max: float, 归一化后的最大值（默认 100）
    - b: float, 指数放大系数（默认 2）

    返回:
    - deformation_percentages: list, 归一化并放大的区域形变量占比
    """

    # 读取已有数据，存入字典 {file_name: deformation_percentages}
    data_dict = {}
    if os.path.exists(outputfile_region):
        with open(outputfile_region, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 5:  # 确保格式正确
                    data_dict[parts[0]] = list(map(float, parts[1:]))

    # 计算形变量
    region_deformations = []
    for region in range(4):
        mask = (region_labels == region)
        if np.sum(mask) == 0:
            region_deformations.append(0)
            continue
        current_region_points = current_points[mask]
        original_region_points = original_points[mask]
        deformation = np.mean(np.linalg.norm(current_region_points - original_region_points, axis=1))
        region_deformations.append(deformation)

    # 转换为 NumPy 数组
    region_deformations = np.array(region_deformations)

    # 归一化到 [0,1]
    if np.max(region_deformations) > 0:
        normalized_deformations = (region_deformations - np.min(region_deformations)) / (np.max(region_deformations) - np.min(region_deformations))
    else:
        normalized_deformations = region_deformations

    # 指数放大
    exp_deformations = np.exp(b * normalized_deformations)

    # 归一化到 [0, scale_max]
    scaled_exp_deformations = (exp_deformations / np.max(exp_deformations)) * scale_max if np.max(exp_deformations) > 0 else exp_deformations

    # 计算形变占比
    total_deformation = np.sum(scaled_exp_deformations)
    if total_deformation > 0:
        deformation_percentages = scaled_exp_deformations / total_deformation
    else:
        deformation_percentages = np.zeros_like(scaled_exp_deformations)

    # 更新数据字典，覆盖旧数据
    data_dict[file_name] = deformation_percentages.tolist()

    # 重新写入所有数据（覆盖文件）
    with open(outputfile_region, "w") as f:
        for key, values in data_dict.items():
            f.write(f"{key}," + ",".join(map(str, values)) + "\n")

    return deformation_percentages.tolist()


# 初始化点云的方法，基于图像的二值面积
def initialize_points_from_binary_area(area: torch.Tensor, n_points: int) -> torch.Tensor:
    sidelength = area.shape[0]
    points = []
    
    while len(points) < n_points:
        random_points = torch.FloatTensor(n_points, 2).uniform_(0, sidelength-1)
        idx = torch.round(random_points).long()
        valid_points = area[idx[:, 0], idx[:, 1]] == 1
        random_points = random_points[valid_points]
        
        if len(points) > 0:
            points = torch.cat([points, random_points], dim=0)
        else:
            points = random_points
            
    return points[:n_points]

# 读取A的点云图像
def load_image_as_point_cloud(image_path, num_points):
    image = Image.open(image_path).convert("L")
    image = np.array(image)
    area = np.where(image > 128, 1, 0)
    area_tensor = torch.tensor(area, dtype=torch.float32)
    points = initialize_points_from_binary_area(area_tensor, num_points)
    return points.unsqueeze(0)  # 返回形状为 (1, num_points, 2) 的点云

def compute_point_cloud_mapping_gpu_2(points_A, points_B, max_matches=2):
    # 将输入张量移动到CPU并转换为NumPy数组
    points_A_np = points_A.cpu().numpy() if torch.is_tensor(points_A) else points_A
    points_B_np = points_B.cpu().numpy() if torch.is_tensor(points_B) else points_B
    
    n_A = points_A_np.shape[0]
    n_B = points_B_np.shape[0]
    
    # 计算距离矩阵
    dist_matrix = np.linalg.norm(points_A_np[:, np.newaxis] - points_B_np, axis=2)
    
    # 初始化匹配计数和结果
    count_B = np.zeros(n_B, dtype=int)
    matches = -np.ones(n_A, dtype=int)
    b_preference = np.zeros(n_B, dtype=float)
    
    # 贪心匹配逻辑
    for i in range(n_A):
        sorted_b_indices = np.argsort(dist_matrix[i, :])
        total_distances = dist_matrix[i, sorted_b_indices] + b_preference[sorted_b_indices]
        
        for b_idx in sorted_b_indices[np.argsort(total_distances)]:
            if count_B[b_idx] < max_matches:
                matches[i] = b_idx
                count_B[b_idx] += 1
                b_preference[b_idx] += 30
                break
    
    return torch.tensor(matches)


# 处理每张图片并生成形变分布图
def process_image_with_plot(file_name, template_points,region_labels, outputfile_region,output_all):
    image_path = os.path.join(particles_path, file_name)
    B_points = load_image_as_point_cloud(image_path, num_points=num_point).to(device)
    indices = compute_point_cloud_mapping_gpu_2(template_points.squeeze(0), B_points.squeeze(0))
    B_points = B_points[:, indices, :]
    
    # 计算每个点的形变大小
    point_deformations = torch.norm(B_points - template_points, dim=2).squeeze(0).cpu().numpy()
    total_deformation = point_deformations.mean()
    
    # 计算每个区域的形变占比
    region_deformations = calculate_region_deformation(
        file_name,
        B_points.squeeze(0).detach().cpu().numpy(),
        template_points.squeeze(0).detach().cpu().numpy(),
        region_labels,
        outputfile_region
    )

    # 更新模板形变量
    updated_deformation = update_template_deformation(
        file_name,
        B_points,
        template_points,
        region_labels,
        deformation_accumulator,
        count_accumulator
    )

    # 可视化模板形变量
    plot_deformation_with_template(template_points, updated_deformation,output_all)

    return file_name, total_deformation


# 修改主函数，处理每张图片并保存图像
def main():
    # 加载模板点云
    template_points = load_image_as_point_cloud(template_path, num_points=num_point).to(device)
    
    region_labels = divide_points_into_quadrants(template_points.squeeze(0).cpu().numpy(), img_width, img_height)

    # 获取particles_path目录下所有png文件
    image_files = [f for f in os.listdir(particles_path) if f.endswith('.png')]
    
    # 存储结果
    results = []
    for file_name in tqdm(image_files, desc="Processing Images"):
        file_name, deformation = process_image_with_plot(file_name, template_points, region_labels, outputfile_region,output_all)
        results.append((file_name, deformation))
    
        # 按deformation值降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 保存到txt文件
        with open(output_file_path, 'w') as output_file:
            for file_name, deformation in results:
                output_file.write(f"{file_name} {deformation}\n")

if __name__ == "__main__":
    main()
