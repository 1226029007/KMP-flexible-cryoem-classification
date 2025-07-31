import os
import cv2
import numpy as np
from tqdm import tqdm

# 文件路径定义
pic_binary_dir = "/home/mozhengao/code/particle/data5_10796/pic_binary"
pic_contour_dir = "/home/mozhengao/code/particle/data5_10796/pic_contour"
txt_dir = "/home/mozhengao/code/particle/data5_10796/pic_with_txt"

# 输出路径
output_binary_dir = "/home/mozhengao/code/particle/data5_10796/processed_pic_binary"
output_contour_dir = "/home/mozhengao/code/particle/data5_10796/processed_pic_contour"

# 创建输出文件夹
os.makedirs(output_binary_dir, exist_ok=True)
os.makedirs(output_contour_dir, exist_ok=True)

def rotate_and_translate_image(image, angle, tx, ty):
    """
    对图像进行旋转和平移。
    
    参数：
        image (np.ndarray): 输入图像。
        angle (float): 旋转角度（以度为单位）。
        tx (float): 水平平移量。
        ty (float): 垂直平移量。
    
    返回：
        np.ndarray: 处理后的图像。
    """
    # 获取图像中心
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 构造旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 叠加平移
    rotation_matrix[0, 2] += tx
    rotation_matrix[1, 2] += ty
    
    # 应用仿射变换
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return rotated_image

# 读取txt文件并处理图像
txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]

for txt_file in tqdm(txt_files, desc="Processing datasets", unit="file"):
    txt_path = os.path.join(txt_dir, txt_file)
    
    with open(txt_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            
            # 获取图片信息和参数
            image_name = parts[0]
            angle = float(parts[1])
            tx = float(parts[2])
            ty = float(parts[3])
            
            # 定位每个文件夹中的图片路径
            binary_path = os.path.join(pic_binary_dir, image_name)
            contour_path = os.path.join(pic_contour_dir, image_name)
            
            # 确保图片存在
            if not all(os.path.exists(p) for p in [binary_path, contour_path]):
                continue
            
            # 读取图片
            binary_image = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
            contour_image = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
            
            # 对图片进行旋转和平移
            processed_binary = rotate_and_translate_image(binary_image, angle, tx, ty)
            processed_contour = rotate_and_translate_image(contour_image, angle, tx, ty)
            
            # 保存处理后的图片
            cv2.imwrite(os.path.join(output_binary_dir, image_name), processed_binary)
            cv2.imwrite(os.path.join(output_contour_dir, image_name), processed_contour)

print("Processing complete!")
