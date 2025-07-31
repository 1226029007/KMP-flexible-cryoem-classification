#对颗粒进行特征提取
import cv2
import numpy as np
import os
from tqdm import tqdm  # 导入tqdm库

# 步骤 1：处理图像并提取最大连通区域
def process_image(image_path: str, percentage: float = 0.3, min_area: int = 2000) -> np.ndarray:
    """
    读取图像并提取面积最大的连通区域，阈值由像素强度的前百分比计算得出。

    参数：
        image_path (str): 图像文件路径。
        percentage (float): 用于计算阈值的百分比（0.0 到 1.0）。
        min_area (int): 连通区域最小面积，排除噪声。

    返回：
        np.ndarray: 处理后的图像，只保留最大连通区域。
    """
    # 读取图像为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # image = 255-image
    # 根据像素值的百分比计算阈值
    intensity_threshold = int(np.percentile(image, (1 - percentage) * 100))
    
    # 二值化处理，生成二值图像
    _, binary = cv2.threshold(image, intensity_threshold, 255, cv2.THRESH_BINARY)

    # 查找连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=4)

    # 创建掩码，只保留最大连通区域
    mask = np.zeros_like(binary)
    max_area = 0
    max_label = 0
    for i in range(1, num_labels):  # 排除背景标签 0
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_label = i

    # 只保留最大连通区域
    mask[labels == max_label] = 1

    # 使用掩码过滤图像，保留最大连通区域
    filtered_image = image * mask

    return image, binary, filtered_image, mask  # 返回原图、二值图、过滤后的图像和掩码


# 步骤 2：提取并保存面积最大的轮廓
def extract_and_save_largest_contour(filtered_image: np.ndarray, output_path: str) -> None:
    """
    提取并保存最大面积的轮廓。

    参数：
        filtered_image (np.ndarray): 处理后的图像，仅包含最大连通区域。
        output_path (str): 保存轮廓图像的路径。
    """
    # 提取轮廓
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算并选择最大轮廓
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # 创建空白图像用于绘制最大轮廓
    contour_image = np.zeros_like(filtered_image)

    # 绘制最大轮廓
    if max_contour is not None:
        cv2.drawContours(contour_image, [max_contour], -1, (255), 1)

    # 保存轮廓图像
    cv2.imwrite(output_path, contour_image)

# 步骤 3：处理并保存图像（包括最大连通区域、二值化图和轮廓）
def process_and_save_image(image_path: str, output_filtered_image_path: str, output_contour_image_path: str,
                           output_binary_image_path: str, intensity_threshold: float = 0.35, min_area: int = 2000) -> None:
    """
    处理图像并保存最大连通区域、二值化图像和最大轮廓。

    参数：
        image_path (str): 输入图像路径。
        output_filtered_image_path (str): 处理后的图像（只保留最大连通区域）的保存路径。
        output_contour_image_path (str): 提取的最大轮廓图像的保存路径。
        output_binary_image_path (str): 二值化图像保存路径。
        intensity_threshold (int): 二值化的阈值。
        min_area (int): 连通区域最小面积。
    """
    # 步骤 1：提取最大连通区域
    original_image, binary_image, filtered_image, mask = process_image(image_path, intensity_threshold, min_area)

    # 步骤 2：保存二值化图像，仅保留最大连通区域
    binary_image_filtered = binary_image * mask  # 仅保留最大连通区域
    cv2.imwrite(output_binary_image_path, binary_image_filtered)

    # 步骤 3：保存最大连通区域的图像
    cv2.imwrite(output_filtered_image_path, filtered_image)

    # 步骤 4：提取并保存最大轮廓
    extract_and_save_largest_contour(filtered_image, output_contour_image_path)

    # print(f"Binary image saved at: {output_binary_image_path}")
    # print(f"Filtered image saved at: {output_filtered_image_path}")
    # print(f"Largest contour image saved at: {output_contour_image_path}")

# 主程序
def main():
    # 输入图像所在目录和输出目录
    input_dir = '/home/mozhengao/code/particle/data2_simulate/pic'
    output_filtered_dir = '/home/mozhengao/code/particle/data2_simulate/filtered'
    output_contour_dir = '/home/mozhengao/code/particle/data2_simulate/contour'
    output_binary_dir = '/home/mozhengao/code/particle/data2_simulate/binary'

    # 创建输出目录（如果不存在）
    os.makedirs(output_filtered_dir, exist_ok=True)
    os.makedirs(output_contour_dir, exist_ok=True)
    os.makedirs(output_binary_dir, exist_ok=True)

    # 获取输入目录下所有PNG文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    # 对每个图像文件进行处理，并添加进度条
    for image_file in tqdm(image_files, desc="Processing images", unit="image"):
        image_path = os.path.join(input_dir, image_file)
        
        output_filtered_image_path = os.path.join(output_filtered_dir, image_file)
        output_contour_image_path = os.path.join(output_contour_dir, image_file)
        output_binary_image_path = os.path.join(output_binary_dir, image_file)

        # 处理并保存图像
        process_and_save_image(image_path, output_filtered_image_path, output_contour_image_path, output_binary_image_path)

# 运行主程序
if __name__ == "__main__":
    main()
