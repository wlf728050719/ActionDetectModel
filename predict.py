import torch
import numpy as np
import pandas as pd
from torch import nn
from net.st_gcn import Model
import json
import argparse


class SkeletonPredictor:
    def __init__(self, config_path, model_path,layout,strategy,embed_dim,max_frames,device=None):
        """
        骨骼动作识别预测器 (带中心距离计算)

        参数:
            config_path: 配置文件路径
            model_path: 模型权重路径
            device: 指定设备 (None则自动选择)
        """
        # 加载配置
        with open(config_path) as f:
            self.config = json.load(f)

        # 设备设置
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型参数
        self.in_channels = 2  # (x,y)坐标
        self.num_class = len(self.config['actions'])
        self.embed_dim = embed_dim  # 必须与训练时一致
        self.graph_args = {'layout': layout, 'strategy': strategy}
        self.edge_importance_weighting = True
        self.max_frames = max_frames  # 必须与训练时一致

        # 初始化模型
        self.model = Model(
            self.in_channels,
            self.embed_dim,  # 输出嵌入维度而非类别数
            self.graph_args,
            self.edge_importance_weighting
        ).to(self.device)

        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 加载中心点
        if 'center_state_dict' in checkpoint:
            self.centers = checkpoint['center_state_dict']['centers'].to(self.device)
        else:
            raise AttributeError('Center state dict not found.')

        # 创建动作标签映射
        self.action_map = {i: action['name'] for i, action in enumerate(self.config['actions'])}
        print("动作标签映射:", self.action_map)

    def preprocess(self, csv_path):
        """
        预处理CSV文件为模型输入格式

        参数:
            csv_path: 输入的CSV文件路径

        返回:
            预处理后的张量 (形状 [1, 2, T, V, 1])
        """
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        num_frames = min(len(df), self.max_frames)

        # 提取x,y坐标 (忽略置信度)
        x_coords = df.filter(regex='kp_.*_x').values[:self.max_frames]
        y_coords = df.filter(regex='kp_.*_y').values[:self.max_frames]

        # 组合成形状为 [2, T, V] 的数组 (T=时间步, V=关键点)
        skeleton_data = np.stack([x_coords, y_coords], axis=0)  # [2, T, V]

        # 如果帧数不足max_frames，则补零
        if skeleton_data.shape[1] < self.max_frames:
            pad_width = ((0, 0), (0, self.max_frames - skeleton_data.shape[1]), (0, 0))
            skeleton_data = np.pad(skeleton_data, pad_width, mode='constant')

        # 转换为float32并添加batch和通道维度
        skeleton_data = torch.from_numpy(skeleton_data.astype(np.float32)).unsqueeze(0).unsqueeze(-1)

        return skeleton_data.to(self.device)

    def predict(self, csv_path):
        """
        预测CSV文件中的动作类别，并计算与各类中心的距离

        参数:
            csv_path: 输入的CSV文件路径

        返回:
            dict: 包含预测结果和距离信息
        """
        # 预处理数据
        input_data = self.preprocess(csv_path)

        # 获取嵌入向量
        with torch.no_grad():
            embedding = self.model(input_data)  # 形状: [1, embed_dim]

            # 计算与所有中心的距离
            distances = torch.cdist(embedding, self.centers)  # 形状: [1, num_class]
            distances = distances.squeeze(0)  # 形状: [num_class]

            # 找到最近的中心 (预测类别)
            min_dist, pred_class = torch.min(distances, 0)

            # 计算softmax概率 (距离越近概率越高)
            probabilities = nn.functional.softmax(-distances, dim=0)

        # 转换为Python标量
        pred_class = pred_class.item()
        min_dist = min_dist.item()
        distances = distances.cpu().numpy()
        probabilities = probabilities.cpu().numpy()

        # 准备距离和概率的详细字典
        class_distances = {}
        class_probabilities = {}
        for class_id in range(self.num_class):
            class_name = self.action_map.get(class_id, f"Class_{class_id}")
            class_distances[class_id] = {
                'distance': float(distances[class_id]),
                'name': class_name
            }
            class_probabilities[class_id] = {
                'probability': float(probabilities[class_id]),
                'name': class_name
            }

        return {
            'predicted_class': pred_class,
            'predicted_class_name': self.action_map.get(pred_class, f"Class_{pred_class}"),
            'min_distance': min_dist,
            'class_distances': class_distances,
            'class_probabilities': class_probabilities,
            'embedding': embedding.cpu().numpy()[0]  # 返回嵌入向量
        }


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='骨骼动作识别预测(带中心距离)')
    parser.add_argument('--csv_file', type=str,required=True, help='输入的CSV文件路径')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--model', type=str, default=r'checkpoints/yolopose/spatial/best_model.pth', help='模型权重路径')
    parser.add_argument('--layout', type=str, default='yolopose', help='layout of the dataset: openpose, ntu-rgb+d, ntu_edge, yolopose')
    parser.add_argument('--strategy', type=str, default='spatial', help='strategy of training: uniform, distance, spatial')
    parser.add_argument('--embed_dim', type=int, default=4096, help='embed_dim')
    parser.add_argument('--max_frames', type=int, default=50, help='max_frames')
    args = parser.parse_args()

    CONFIG = args.config
    CSV_FILE = args.csv_file
    LAYOUT = args.layout
    STRATEGY = args.strategy
    EMBED_DIM = args.embed_dim
    MAX_FRAMES = args.max_frames
    MODEL = args.model
    # 初始化预测器
    predictor = SkeletonPredictor(CONFIG, MODEL, LAYOUT, STRATEGY, EMBED_DIM, MAX_FRAMES)

    # 执行预测
    try:
        result = predictor.predict(CSV_FILE)
        print("\n预测结果:")
        print(f"预测类别ID: {result['predicted_class']}")
        print(f"预测类别名称: {result['predicted_class_name']}")
        print(f"与最近中心的距离: {result['min_distance']:.4f}")

        # 打印所有类别的距离
        print("\n与各类中心的距离:")
        for class_id, info in result['class_distances'].items():
            print(f"{class_id}: {info['name']} - 距离: {info['distance']:.4f}")

        # 打印所有类别的概率
        print("\n各类别概率:")
        for class_id, info in result['class_probabilities'].items():
            print(f"{class_id}: {info['name']} - 概率: {info['probability']:.2%}")

    except Exception as e:
        print(f"预测失败: {str(e)}")