import os
import numpy as np

# 收单个文件夹作为输入，读取其中所有 .npy 文件并计算每个文件中总和，最后返回这些总和的平均值。
def calculate_average_summation(folder):
    # 获取文件夹中的npy文件列表，并按文件名排序
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    
    # 初始化变量存储非零元素平均值的总和和文件个数
    total_mean = 0
    file_count = len(files)
    
    if file_count == 0:
        raise ValueError("文件夹中没有.npy文件")

    # 遍历文件，读取数组并计算非零元素的平均值
    for file in files:
        # 构建文件的完整路径
        file_path = os.path.join(folder, file)
        
        # 读取npy文件为数组
        arr = np.load(file_path)
        
        # 计算非零元素的平均值
        non_zero_elements = arr[arr != 0]
        if non_zero_elements.size == 0:
            mean_value = 0
        else:
            mean_value = np.sum(non_zero_elements)
        
        # 累加平均值
        total_mean += mean_value
    
    # 计算所有文件的平均非零值
    average_mean = total_mean / file_count
    return average_mean

import matplotlib.pyplot as plt
if __name__ == "__main__":
    root_folder = '/home/ubuntu/Project/MT-Net/init_400_infer'  # 替换为存放多个子文件夹的根文件夹路径
    save_path = '/home/ubuntu/Project/MT-Net/ratio_images'  # 替换为保存图像的路径
    os.makedirs(save_path, exist_ok=True)  # 确保保存路径存在

    subfolders = ['sccsf_error', 'sccsf_uncertainty', 'sc_error', 'sc_uncertainty']
    colors = ['b', 'g', 'r', 'c']
    labels = ['SCCSF Error', 'SCCSF Uncertainty', 'SC Error', 'SC Uncertainty']
    
    root_folder = '/home/ubuntu/Project/UI-MAE/variance/onestage_wopretrain'  # 替换为存放多个子文件夹的根文件夹路径
    save_path = '/home/ubuntu/Project/UI-MAE/ratio_images/onestage_wopretrain'  # 替换为保存图像的路径
    os.makedirs(save_path, exist_ok=True)  # 确保保存路径存在

    subfolders = ['error', 'variance']
    colors = ['b', 'g']
    labels = ['Error', 'Uncertainty']

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for subfolder, color, label in zip(subfolders, colors, labels):
        training_steps = []
        average_values = []

        # 遍历根文件夹中的所有子文件夹
        for folder in os.listdir(root_folder):
            subfolder_path = os.path.join(root_folder, folder, subfolder)

            if os.path.isdir(subfolder_path):
                try:
                    # 从文件夹名称中提取训练轮数
                    # 文件夹名格式：10MAE, 100MAE 等
                    training_step = int(''.join(filter(str.isdigit, folder)))  # 提取数字部分作为训练轮数
                    training_steps.append(training_step)

                    # 计算当前子文件夹中所有npy文件的平均总和
                    avg_value = calculate_average_summation(subfolder_path)
                    average_values.append(avg_value)

                    print(f"Subfolder: {folder}, Training Steps: {training_step}, Average Non-Zero Value: {avg_value:.4f}")

                except Exception as e:
                    print(f"Skipping folder {folder}: {e}")

        # 确保训练轮数和平均值按照训练轮数排序
        if training_steps and average_values:
            sorted_training_steps, sorted_average_values = zip(*sorted(zip(training_steps, average_values)))

            # 绘制训练轮数和平均值的曲线
            #if 'uncertainty' in subfolder:
            if 'variance' in subfolder:
                ax1.plot(sorted_training_steps, sorted_average_values, marker='o', linestyle='-', color=color, label=label)
            else:
                ax2.plot(sorted_training_steps, sorted_average_values, marker='o', linestyle='-', color=color, label=label)

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Average Uncertainty")
    ax2.set_ylabel("Average Error")
    plt.title("Training Steps vs. Average Values")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    ax1.grid(True)

    # 保存图像到指定路径
    plt.savefig(os.path.join(save_path, "training_vs_average_values.png"), dpi=300)
    plt.show()