from tqdm import tqdm  # 导入 tqdm 库
import os
import mrcfile
import matplotlib.pyplot as plt
import numpy as np

def read_star_file(star_file_path):
    """
    读取 .star 文件，提取有效颗粒的数据。
    """
    with open(star_file_path, 'r') as file:
        lines = file.readlines()

    # 忽略前 23 行
    particle_lines = lines[23:]

    # 判断有效的颗粒数据行（必须包含 17 列）
    valid_particles = []
    for line in particle_lines:
        cols = line.split()
        if len(cols) == 17:
            # 提取旋转和平移信息
            anglePsi = float(cols[2])  # 第3列为 _rlnAnglePsi
            originXAngst = float(cols[10])  # 第11列为 _rlnOriginXAngst
            originYAngst = float(cols[11])  # 第12列为 _rlnOriginYAngst
            X = float(cols[0])
            Y = float(cols[1])
            valid_particles.append((anglePsi, originXAngst, originYAngst, X, Y))

    # 返回有效颗粒的数据
    return valid_particles

def load_total_particles(total_star_path):
    """
    读取总的颗粒 .star 文件，提取坐标和角度信息。
    """
    with open(total_star_path, 'r') as file:
        lines = file.readlines()

    # 忽略前 23 行
    particle_lines = lines[41:]

    # 存储总颗粒信息
    total_particles = []
    for line in particle_lines:
        cols = line.split()
        if len(cols) >= 3:  # 确保至少有 _rlnCoordinateX, _rlnCoordinateY, _rlnAnglePsi
            x = float(cols[0])  # _rlnCoordinateX
            y = float(cols[1])  # _rlnCoordinateY
            anglePsi = float(cols[2])  # _rlnAnglePsi
            total_particles.append((x, y, anglePsi))

    return total_particles

def find_particle_index(particles_data, total_particles):
    """
    根据 (X, Y, anglePsi) 匹配颗粒，并找到其在总颗粒文件中的行号。
    """
    indices = []
    for particle in particles_data:
        anglePsi, _, _, x, y = particle
        try:
            index = total_particles.index((x, y, anglePsi))
            indices.append(index + 42)  # 加上前 23 行和 0 索引
        except ValueError:
            indices.append(-1)  # 未匹配到时返回 -1
    return indices

def export_mrcs_to_png_and_txt(input_mrcs_path, output_folder, particles_data, base_filename, total_particles, all_txt_path):
    """
    将 .mrcs 文件中的每张图像保存为 .png，并记录对应的 .txt 文件。
    同时，将颗粒名称和对应的总颗粒行数保存到 all.txt。
    """
    txt_file_path = os.path.join(output_folder, f"{base_filename}.txt")
    with open(txt_file_path, 'w') as txt_file, open(all_txt_path, 'a') as all_txt_file:
        with mrcfile.open(input_mrcs_path, permissive=True) as mrc:
            data = mrc.data

            # 匹配总颗粒的行号
            indices = find_particle_index(particles_data, total_particles)

            # 遍历每个颗粒
            for idx, ((anglePsi, originXAngst, originYAngst, X, Y), index) in enumerate(zip(particles_data, indices), start=1):
                # 获取颗粒图像
                if data.ndim == 2:
                    selected_image = data
                else:
                    selected_image = data[idx - 1, :, :]  # idx 从 1 开始，而数组是 0 索引

                # 保存颗粒图像为 PNG
                png_filename = f"{base_filename}_{idx}.png"
                output_png_path = os.path.join(output_folder, png_filename)
                plt.imsave(output_png_path, selected_image, cmap='gray')

                # 写入 .txt 文件信息，第一列记录 .png 文件的名字
                txt_file.write(f"{png_filename} {anglePsi} {originXAngst} {originYAngst} {X} {Y}\n")

                # 写入 all.txt 文件
                if index != -1:
                    all_txt_file.write(f"{png_filename} {index}\n")

def process_directory(input_directory, output_folder, total_star_path, all_txt_path):
    """
    处理 input_directory 下的所有 .mrcs 文件和对应的 .star 文件，提取颗粒并保存为 .png，并记录 .txt 文件。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载总颗粒文件信息
    total_particles = load_total_particles(total_star_path)

    # 获取所有文件
    files = [f for f in os.listdir(input_directory) if f.endswith('.star')]

    # 清空 all.txt 文件
    with open(all_txt_path, 'w') as all_txt_file:
        all_txt_file.write('')

    # 进度条显示整个目录的处理进度
    for file in tqdm(files, desc="Processing directory"):
        star_file_path = os.path.join(input_directory, file)
        mrcs_file_path = star_file_path.replace('_extract.star', '.mrcs')

        if not os.path.exists(mrcs_file_path):
            print(f"Skipping {star_file_path}: corresponding .mrcs file not found.")
            continue

        # 获取文件的基本名称（去掉扩展名）
        base_filename = os.path.splitext(os.path.basename(mrcs_file_path))[0]

        # 读取颗粒数据
        particles_data = read_star_file(star_file_path)

        # 导出 PNG 和记录信息
        export_mrcs_to_png_and_txt(mrcs_file_path, output_folder, particles_data, base_filename, total_particles, all_txt_path)

# 设置输入目录、输出目录和总颗粒文件路径
input_directory = '/home/mozhengao/code/relion/Extract/job158/10706_z2'
total_star_path = '/home/mozhengao/code/relion/Extract/job158/particles.star'
output_folder = '/home/mozhengao/code/particle/data5_10796/pic_with_txt'
all_txt_path = '/home/mozhengao/code/particle/data5_10796/all.txt'

# 执行目录处理
process_directory(input_directory, output_folder, total_star_path, all_txt_path)
