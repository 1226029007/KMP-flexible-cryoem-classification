KMP Flexible Cryo-EM Particle Classification 使用说明
------------------------------------------------------------------------------------------
简介

此项目用于颗粒的形变分类，主要通过多种数据处理和算法模型来提取、对齐、可视化和分析颗粒形变情况。项目包括了一系列 Python 脚本，每个脚本实现不同的功能，涉及从 Relion 提取的数据处理，到颗粒形变检测和可视化等任务。
------------------------------------------------------------------------------------------
依赖

请确保安装了以下依赖项：

Python 3.x

相关的 Python 库：numpy, matplotlib, scipy, requests, opencv-python 等
------------------------------------------------------------------------------------------
脚本说明

1_extract.py
功能：将 Relion 中提取出的 .mrcs 图像拆分成 .png 格式图片，并保存对应的对齐参数。

输入：

input_directory: Relion 中提取的 .mrcs 和 .star 文件路径

total_star_path: Relion 中的 particles.star 文件路径

输出：

output_folder: 每个颗粒的 .png 图片和其对齐参数 .txt

all_txt_path: 颗粒的索引列表，用于后续的形变颗粒剔除

2_data_make.py
功能：用于制作颗粒的二值化和轮廓图片。

输入：

input_dir: 导入的图片路径（由 1_extract.py 生成）

输出：

output_contour_dir: 颗粒轮廓数据集路径

output_binary_dir: 颗粒二值化数据集路径

3_data_al.py
功能：用于颗粒与模板的对齐。

输入：

pic_binary_dir: 二值化数据集（由 2_data_make.py 生成）

pic_contour_dir: 轮廓数据集（由 2_data_make.py 生成）

txt_dir: 输入的对齐参数路径（由 1_extract.py 生成）

输出：

output_binary_dir: 对齐后二值化数据集路径

output_contour_dir: 对齐后的轮廓数据集路径

4_single.py
功能：计算单个颗粒与模板的形变过程并可视化。

输入：

particles_path: 颗粒路径

template_path: 模板路径

输出：

output_path: 可视化结果（形变过程、匹配结果、loss、形变情况）

4_batch.py
功能：批量计算颗粒相较于模板的形变量，并生成相关形变数据。

输入：

particles_path: 颗粒路径

template_path: 模板路径

输出：

output_file_path: 形变值降序表

outputfile_region: 颗粒形变区域情况

output_all: 模板的形变情况均值

5_visiable.py
功能：查看形变颗粒的图片。

输入：

input_txt_path: 颗粒形变降序 .txt 文件（由 4_deformation.py 生成）

source_image_dir: 数据集路径（由 1_extract.py 生成）

输出：

output_dir: 导出形变最大数目的颗粒图片

6_metric.py
功能：查看评估指标（准确率和召回率）。

输入：

predicted_file: 颗粒形变降序 .txt 文件（由 4_deformation.py 生成）

ground_truth_file: 输入真实标签 .txt 文件（手动打标，包含形变颗粒的名称）

输出：

output: 准确率和召回率的图表

7_relion.py
功能：获取剔除形变颗粒后的 .star 文件。

输入：

pic_binary_path: 颗粒形变降序 .txt 文件（由 4_deformation.py 生成）

all_file_path: 颗粒的索引列表（由 1_extract.py 生成）

particles_star_path: Relion 中的 particles.star 文件路径

输出：

output_path: 输出剔除形变颗粒后的 .star 文件
------------------------------------------------------------------------------------------
使用流程

颗粒提取：

使用 1_extract.py 从 Relion 中提取颗粒的 .mrcs 图片，并生成相应的对齐参数。

数据制作与对齐：

使用 2_data_make.py 和 3_data_al.py 生成颗粒的二值化和轮廓数据集，并进行颗粒与模板的对齐。

形变计算与可视化：

使用 4_single.py 和 4_batch.py 计算单个颗粒和批量颗粒的形变过程，并生成相关的形变数据和可视化结果。

形变颗粒可视化：

使用 5_visiable.py 查看形变颗粒的图片，并根据形变程度选择最显著的颗粒。

评估与结果分析：

使用 6_metric.py 评估模型的准确率和召回率。

生成最终结果：

使用 7_relion.py 从 Relion 中生成剔除形变颗粒后的 particles.star 文件，用于后续分析。