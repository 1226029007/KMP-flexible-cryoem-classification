#查看形变的颗粒的图片
import os
import shutil

# 文件路径定义
input_txt_path = '/home/mozhengao/code/particle/data5_10796/t.txt'
source_image_dir = '/home/mozhengao/code/particle/data5_10796/pic_with_txt'
output_dir = '/home/mozhengao/code/particle/data5_10796/deformation'

# 确保目标目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取 txt 文件，提取前 350 个图片名字
top_images = []
with open(input_txt_path, 'r') as file:
    for i, line in enumerate(file):
        if i >= 30:  # 只取前 350 行
            break
        parts = line.strip().split()
        if len(parts) >= 2:
            image_name = parts[0]  # 图片名字
            top_images.append(image_name)

# 复制图片到目标文件夹
for image_name in top_images:
    source_path = os.path.join(source_image_dir, image_name)
    destination_path = os.path.join(output_dir, image_name)

    # 检查源文件是否存在
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"已复制: {source_path} -> {destination_path}")
    else:
        print(f"未找到源文件: {source_path}")

print("图片复制完成。")
