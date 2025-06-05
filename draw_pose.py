import os
import csv
import matplotlib.pyplot as plt
import argparse

# 定义骨骼连接关系 (0-based索引)
skeleton = [
    [16, 14], [14, 12], [15, 13], [13, 11], [12, 11], [6, 12],
    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 1],
    [1, 0], [0, 3], [2, 4], [3, 5], [4, 6], [5, 7]
]

# 颜色配置
joint_color = 'red'  # 关节点颜色
bone_color = 'green'  # 骨骼颜色

def plot_skeleton(frame_data, frame_num,output_dir,conf_threshold):
    """绘制单帧骨骼图"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # 提取坐标和置信度
    kp_x = [frame_data[f'kp_{i}_x'] for i in range(17)]
    kp_y = [frame_data[f'kp_{i}_y'] for i in range(17)]
    kp_conf = [frame_data[f'kp_{i}_conf'] for i in range(17)]

    # 绘制关节点
    for i, (x, y, conf) in enumerate(zip(kp_x, kp_y, kp_conf)):
        if conf > conf_threshold:
            ax.scatter(x, y, color=joint_color, s=50)
            ax.text(x, y, str(i), fontsize=8, ha='center', va='bottom')

    # 绘制骨骼
    for connection in skeleton:
        start_joint, end_joint = connection
        conf_start = kp_conf[start_joint]
        conf_end = kp_conf[end_joint]

        if conf_start > conf_threshold and conf_end > conf_threshold:
            x_pair = [kp_x[start_joint], kp_x[end_joint]]
            y_pair = [kp_y[start_joint], kp_y[end_joint]]
            ax.plot(x_pair, y_pair, color=bone_color, linewidth=2)

    # 设置坐标轴
    ax.set_xlim(min(kp_x) - 50, max(kp_x) + 50)
    ax.set_ylim(min(kp_y) - 50, max(kp_y) + 50)
    ax.invert_yaxis()  # 图像坐标系Y轴向下
    ax.set_title(f'Frame {frame_num}')
    ax.set_aspect('equal')

    # 保存图像
    output_path = os.path.join(output_dir, f'frame_{frame_num:04d}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()


def process_csv(csv_file,output_dir,conf_threshold):
    """处理CSV文件"""
    file_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_dir = f'{output_dir}/{file_name}'
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # 转换数据类型
            frame_data = {}
            for key in row:
                if key == 'frame':
                    frame_data[key] = int(row[key])
                elif '_conf' in key:
                    frame_data[key] = float(row[key])
                else:
                    frame_data[key] = float(row[key])

            # 绘制骨骼图
            plot_skeleton(frame_data, i, output_dir, conf_threshold)
            print(f'Processed frame {i}')


if __name__ == '__main__':
    #绘制指定csv文件中符合置信度要求的骨骼节点图片
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str,required=True)
    parser.add_argument('--output_dir', type=str,default='skeleton_frames')
    parser.add_argument('--conf_threshold', type=float, default=0.3)
    args = parser.parse_args()
    csv_path = args.csv_path
    output_dir = args.output_dir
    conf_threshold = args.conf_threshold
    process_csv(csv_path,output_dir,conf_threshold)
    print(f"All frames saved to {output_dir}")