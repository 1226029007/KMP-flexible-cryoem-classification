import os

# 文件路径设置
pic_binary_path = '/home/mozhengao/code/particle/data5_10796/t.txt'
all_file_path = '/home/mozhengao/code/particle/data5_10796/all.txt'
particles_star_path = '/home/mozhengao/code/relion/Extract/job158/particles.star'
output_path = '/home/mozhengao/code/particle/data5_10796/filtered_particles9.star'

def read_pic_binary(file_path):
    """
    读取包含颗粒文件名字和形变量的文件。
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip().split() for line in lines]

def read_all_file(all_file_path):
    """
    读取 all.txt 文件，返回颗粒文件名字和行数。
    """
    with open(all_file_path, 'r') as file:
        lines = file.readlines()
    return {line.split()[0]: int(line.split()[1]) for line in lines}

def read_particles_star(star_file_path):
    """
    读取完整的 .star 文件。
    """
    with open(star_file_path, 'r') as file:
        lines = file.readlines()
    return lines

def write_star_file(output_path, data):
    """
    写入新的 .star 文件。
    """
    with open(output_path, 'w') as file:
        file.writelines(data)

def filter_particles(pic_binary_path, all_file_path, particles_star_path, output_path, remove_ratio=0.1):
    """
    根据形变量剔除颗粒，并保存新的 .star 文件。
    """
    # 读取文件
    pic_binary = read_pic_binary(pic_binary_path)
    all_data = read_all_file(all_file_path)
    particles_data = read_particles_star(particles_star_path)

    # 计算需要移除的颗粒数量
    total_particles = len(pic_binary)
    remove_count = int(total_particles * remove_ratio)
    # 找出需要剔除的颗粒名字
    particles_to_remove = [pic_binary[i][0] for i in range(remove_count)]

    # 找到对应的行号
    lines_to_remove = {all_data[particle] for particle in particles_to_remove if particle in all_data}

    # 过滤数据
    filtered_data = [line for idx, line in enumerate(particles_data, start=1) if idx not in lines_to_remove]

    # 写入新的 .star 文件
    write_star_file(output_path, filtered_data)

# 执行过滤
filter_particles(pic_binary_path, all_file_path, particles_star_path, output_path)
