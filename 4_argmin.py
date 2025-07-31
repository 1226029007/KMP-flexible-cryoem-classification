#只处理两个的
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy.optimize import linear_sum_assignment
import cv2
import time

# 记录起始时间
start_time = time.time()

#参数
num_point = 8000
total_epochs = 200  # 训练轮数
LR=0.0001
scheduler_step=100   # 每 scheduler_step 个 epoch 将学习率减半
End_threshold=0.3
template_path = r"/home/mozhengao/code/particle/data2_simulate/10345/particle_binary.png"
particles_path = r"/home/mozhengao/code/particle/data2_simulate/pic_part/Top_Left/j97_Top_Left_04.png"
output_deformation_path = '/home/mozhengao/code/particle/data2_simulate/deformation'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义网络结构
class ShapeDeformationNN(nn.Module):
    def __init__(self, num_points, latent_dim=128):
        super(ShapeDeformationNN, self).__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        
        # 编码器：提取点云的特征
        self.encoder = nn.Sequential(
            nn.Linear(2 * num_points, 512),  # 输入为二维点云 (N, 2)，展开为一维 (2*N, )
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        
        # 解码器：生成每个点的偏移量
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * num_points)  # 输出为每个点的 (x, y) 偏移量
        )

    def forward(self, x):
        # 输入为 (batch_size, num_points, 2)，展开为一维输入
        x = x.reshape(x.size(0), -1)
        
        # 编码器部分
        z = self.encoder(x)
        
        # 解码器部分
        displacement = self.decoder(z)
        
        # 将输出恢复为每个点的位移 (batch_size, num_points, 2)
        displacement = displacement.reshape(-1, self.num_points, 2)
        return displacement


# 定义损失函数
def loss_function(predicted_displacement, target_points, current_points, num_points, indices):
    
    # 计算B点云的变形（基于最优匹配） 
    new_points = current_points + predicted_displacement
    dist = torch.norm(new_points - target_points, dim=2)  # 计算每个点的欧式距离
    loss = dist.sum() / num_points  # 对所有点的距离求和
    return loss


# 保存B点云的函数
def save_point_cloud_image(points, epoch):
    plt.figure(figsize=(img_width / 100, img_height / 100))  # 将图像尺寸缩放为与原始图片相匹配
    plt.scatter(points[:, 1], points[:, 0], c='green', s=0.01)

    # plt.title(f"Point Cloud at Epoch {epoch}")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.axis("equal")
    plt.axis('off')  # 关闭坐标轴
    # plt.xlim(-0.5, 0.5)  # 设置 x 轴范围
    # plt.ylim(-0.5, 0.5)  # 设置 y 轴范围
    plt.xlim(0, img_width)  # 设置 x 轴范围
    plt.ylim(0, img_height)  # 设置 y 轴范围
    plt.gca().invert_yaxis()  # 翻转 y 轴，匹配图像坐标系
    
    plt.savefig(f"/home/mozhengao/code/particle/result/point_cloud_epoch_{epoch}.png", dpi=100)  # 设置 DPI，影响清晰度
    plt.close()



# 初始化点云的方法，基于图像灰度值的加权采样
def initialize_points_from_intensity(image: np.ndarray, n_points: int, intensity_threshold: int = 140, min_area: int = 0) -> torch.Tensor:
    h, w = image.shape
    
    # 应用阈值，保留大于 `intensity_threshold` 的区域
    binary_image = image > intensity_threshold  # 布尔掩码

    # 找到连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image.astype(np.uint8),connectivity=4)

    # 创建一个掩码，只保留大于 `min_area` 的连通区域（排除噪声）
    mask = np.zeros_like(binary_image)
    for i in range(1, num_labels):  # 排除背景标签 0
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 1  # 1 表示保留该连通区域

    # 根据掩码保留颗粒部分，并按强度进行加权
    filtered_image = image * mask  # 保留颗粒部分，噪声部分为 0

    # 将灰度值转换为权重（归一化到 [0, 1]），同时保留强度的连续性
    weights = filtered_image.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    weights[weights < 0.001] = 0  # 对噪声部分赋值为 0，避免采样

    # 归一化为概率分布
    weights /= weights.sum()  # 使得总权重为 1

    # 使用权重矩阵进行加权随机采样
    indices = np.arange(h * w)  # 图像像素的线性索引
    sampled_indices = np.random.choice(indices, size=n_points, p=weights.flatten())
    sampled_coords = np.column_stack(np.unravel_index(sampled_indices, (h, w)))  # 转为坐标

    # 转为 PyTorch 张量
    points = torch.tensor(sampled_coords, dtype=torch.float32)
    return points



# 读取图像并初始化点云
def load_image_as_point_cloud(image_path, num_points):
    image = Image.open(image_path).convert("L")
    image = np.array(image)
    
    points = initialize_points_from_intensity(image, num_points)
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


def save_loss_history_plot(loss_history, save_path="/home/mozhengao/code/particle/result/loss_history.png"):
    plt.figure()
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# 区域划分函数
def divide_points_into_quadrants(points, img_width, img_height):
    region_labels = []
    for point in points:
        x, y = point
        if x < img_width / 2 and y < img_height / 2:
            region_labels.append(0)  # 左上
        elif x >= img_width / 2 and y < img_height / 2:
            region_labels.append(1)  # 右上
        elif x < img_width / 2 and y >= img_height / 2:
            region_labels.append(2)  # 左下
        else:
            region_labels.append(3)  # 右下
    return np.array(region_labels)


# 区域形变计算函数
def calculate_region_deformation(current_points, original_points, region_labels):
    region_deformations = []
    for region in range(4):  # 四个区域
        mask = (region_labels == region)
        if np.sum(mask) == 0:  # 该区域没有点
            region_deformations.append(0)
            continue
        current_region_points = current_points[mask]
        original_region_points = original_points[mask]
        deformation = np.mean(np.linalg.norm(current_region_points - original_region_points, axis=1))
        region_deformations.append(deformation)
    return region_deformations

def plot_matched_point_clouds(A_points, B_points, indices, save_path="/home/mozhengao/code/particle/result/matched_points.png"):
    plt.figure(figsize=(8, 8))

    # 绘制所有A的点云（显示所有点）
    plt.scatter(A_points[:, 1], A_points[:, 0], c='blue', label="Template", s=10)
    # 绘制所有B的点云（显示所有点）
    plt.scatter(B_points[:, 1], B_points[:, 0], c='red', label="Deformed", s=10)

    # 根据indices连接对应的点
    for i in range(len(indices)):
        A_point = A_points[i]
        B_point = B_points[indices[i]]

        # 使用灰色线连接对应点
        plt.plot([A_point[1], B_point[1]], [A_point[0], B_point[0]], c='gray', lw=0.5)

    plt.title("Matched Point Cloud (A vs. B)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()  # 翻转y轴，符合图像坐标系
    plt.axis("equal")
    plt.savefig(save_path)
    plt.close()

def save_binary_point_cloud(points, img_width, img_height, save_path, point_radius=2):
    """
    优化后的点云保存函数：每个点用圆形区域表示，增加覆盖面积以提高IoU准确性
    参数:
        point_radius: 点的半径（像素），控制覆盖区域大小
    """
    # 创建全黑图像（注意OpenCV的尺寸是 (height, width)）
    img = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # 遍历所有点，绘制圆形区域
    for point in points:
        x = int(point[0])
        y = int(point[1])
        # 使用OpenCV绘制实心圆（确保坐标在图像范围内）
        if 0 <= x < img_height and 0 <= y < img_width:
            cv2.circle(
                img,
                center=(y, x),  # OpenCV的坐标顺序为 (x,y)，但此处因图像存储为 (height, width) 需调整
                radius=point_radius,
                color=255,
                thickness=-1  # -1 表示填充
            )
    
    # 保存图像
    Image.fromarray(img).save(save_path)


image = Image.open(template_path)
img_width, img_height = image.size
A_points = load_image_as_point_cloud(template_path, num_points=num_point).to(device)  # 模板A点云
save_point_cloud_image(A_points.squeeze(0).detach().cpu().numpy(), -1)
B_points = load_image_as_point_cloud(particles_path, num_points=num_point).to(device)  # 待测试B点云
save_point_cloud_image(B_points.squeeze(0).detach().cpu().numpy(), 0)



# 计算A和B点云之间的匹配
indices = compute_point_cloud_mapping_gpu_2(A_points.squeeze(0), B_points.squeeze(0))

# 使用 `indices` 绘制匹配关系
plot_matched_point_clouds(A_points.squeeze(0).detach().cpu().numpy(),B_points.squeeze(0).detach().cpu().numpy(), indices)
B_points = B_points[:, indices, :]
tem_B_points_particle = B_points
# 计算按indices对应关系的点云距离之和
total_distance = torch.sum(torch.norm(A_points.squeeze(0) - B_points.squeeze(0), dim=1))/num_point
print(f'Total distance between matched points: {total_distance.item()}')

# 计算区域标签
region_labels = divide_points_into_quadrants(A_points.squeeze(0).cpu().numpy(), img_width, img_height)

# 创建模型并移动到GPU
model = ShapeDeformationNN(num_points=num_point).to(device)

# 优化器
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.5) 

# 训练
deformation_scores = []  # 记录B的型变量

from tqdm import tqdm


loss_history = []  # 用来记录损失值，检查变化趋势
ori_position = B_points
for epoch in tqdm(range(total_epochs), desc="Training Progress"):
    optimizer.zero_grad()
    
    # 获取B点云的位移
    displacement = model(B_points)
    
    # 计算损失
    loss = loss_function(displacement, A_points, B_points, num_points=num_point, indices=indices)
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()

    # 学习率调度器更新
    scheduler.step()

    loss_history.append(loss.item())
    
    current_points = B_points + displacement
    total_deformation = torch.norm(current_points - ori_position, dim=2).sum().item()/num_point
    deformation_scores.append(total_deformation)

    # 计算每个区域的形变占比
    region_deformations = calculate_region_deformation(
        current_points.squeeze(0).detach().cpu().numpy(),
        A_points.squeeze(0).detach().cpu().numpy(),
        region_labels
    )

    # 打印日志
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Total Deformation: {total_deformation}")
        print(f"Region Deformations: {region_deformations}")

    # 每20轮保存一次点云图像
    if (epoch + 1) % 10 == 0:
        save_point_cloud_image(current_points.squeeze(0).detach().cpu().numpy(), epoch + 1)

    # 保存损失历史
    save_loss_history_plot(loss_history)
    
    if loss < End_threshold:
        save_point_cloud_image(current_points.squeeze(0).detach().cpu().numpy(), epoch + 1)
        print("Stop.")
        break

final_deformation = deformation_scores[-1]
print(f'Final deformation (total movement distance) of B relative to A: {final_deformation}')


#绘制形变分布图（包含模板点云）
def plot_deformation_with_template(points, deformations, output_path):
    plt.figure(figsize=(8, 8))
    # 绘制模板点云
    # 根据形变量调整点大小和透明度
    point_sizes = (deformations - deformations.min()) / (deformations.max() - deformations.min()) * 20 + 0.1  # 点大小范围 5 到 25
    point_alphas = (deformations - deformations.min()) / (deformations.max() - deformations.min()) * 0.8 + 0.2  # 透明度范围 0.2 到 1.0
    # 绘制颗粒点云（根据形变大小着色和调整大小/透明度）
    scatter = plt.scatter(
        # points[:, 0], 
        # points[:, 1], 
        points[:, 1],  # 交换 x 和 y
        -points[:, 0],  # 负号让它正确翻转
        c=deformations, 
        cmap="viridis", 
        s=point_sizes, 
        alpha=point_alphas, 
        edgecolors="face",
    )

    plt.axis("equal")
    plt.axis("off")


    # 保存图像
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()


# 计算模板点云的形变大小
point_deformations = torch.norm(tem_B_points_particle - A_points, dim=2).squeeze(0).cpu().numpy()
total_deformation = point_deformations.mean()
plot_deformation_with_template(A_points.squeeze(0).cpu().numpy(),point_deformations,output_deformation_path)

#计算颗粒点云的形变大小
# point_deformations = torch.norm(tem_B_points_particle - B_points, dim=2).squeeze(0).cpu().numpy()
# total_deformation = point_deformations.mean()
# plot_deformation_with_template(tem_B_points_particle.squeeze(0).cpu().numpy(),point_deformations,output_deformation_path)


#计算IOU用，保存二值化图片
# A_points_np = A_points.squeeze(0).detach().cpu().numpy()
# save_binary_point_cloud(
#     A_points_np, 
#     img_width, 
#     img_height, 
#     '/home/mozhengao/code/particle/result/A_points_binary.png',
#     point_radius=3
# )
# B_points_np = B_points + displacement
# B_points_np = B_points_np.squeeze(0).detach().cpu().numpy()
# save_binary_point_cloud(
#     B_points_np, 
#     img_width, 
#     img_height, 
#     '/home/mozhengao/code/particle/result/B_points_binary.png',
#     point_radius=3
# )
# B_points = B_points.squeeze(0).detach().cpu().numpy()
# save_binary_point_cloud(
#     B_points, 
#     img_width, 
#     img_height, 
#     '/home/mozhengao/code/particle/result/B_points_binary_ori.png',
#     point_radius=3
# )


# 记录结束时间
end_time = time.time()

# 计算并打印运行时间
print(f"代码运行时间: {end_time - start_time:.6f} 秒")