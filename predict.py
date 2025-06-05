import torch
import numpy as np
import pandas as pd
from torch import nn
from net.st_gcn import Model
import json
import argparse


class SkeletonPredictor:
    def __init__(self, config_path, model_path, device=None):
        """
        骨骼动作识别预测器

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
        self.graph_args = {'layout': 'yolopose', 'strategy': 'spatial'}
        self.edge_importance_weighting = True
        self.max_frames = 50  # 必须与训练时一致

        # 初始化模型
        self.model = Model(
            self.in_channels,
            self.num_class,
            self.graph_args,
            self.edge_importance_weighting
        ).to(self.device)

        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 创建动作标签映射
        self.action_map = {action['action']: action['name'] for action in self.config['actions']}

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
        预测CSV文件中的动作类别

        参数:
            csv_path: 输入的CSV文件路径

        返回:
            dict: 包含预测结果的信息
        """
        # 预处理数据
        input_data = self.preprocess(csv_path)

        # 预测
        with torch.no_grad():
            outputs = self.model(input_data)
            probabilities = nn.functional.softmax(outputs, dim=1)
            pred_prob, pred_class = torch.max(probabilities, 1)

        # 转换为Python标量
        pred_class = pred_class.item()
        pred_prob = pred_prob.item()

        return {
            'class': pred_class,
            'probability': pred_prob,
            'class_name': self.action_map.get(str(pred_class), 'Unknown'),
            'all_probabilities': probabilities.cpu().numpy()[0]
        }


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='骨骼动作识别预测')
    parser.add_argument('--csv_file', type=str,required=True, help='输入的CSV文件路径')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    args = parser.parse_args()

    # 初始化预测器
    predictor = SkeletonPredictor(args.config, args.model)

    # 执行预测
    try:
        result = predictor.predict(args.csv_file)
        print("\n预测结果:")
        print(f"类别ID: {result['class']}")
        print(f"类别名称: {result['class_name']}")
        print(f"置信度: {result['probability']:.2%}")

        # 打印所有类别的概率
        print("\n所有类别概率:")
        for class_id, prob in enumerate(result['all_probabilities']):
            print(f"{class_id}: {predictor.action_map.get(str(class_id), 'Unknown')} - {prob:.2%}")
    except Exception as e:
        print(f"预测失败: {str(e)}")