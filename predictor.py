import torch
import numpy as np
import pandas as pd
from torch import nn
from net.st_gcn import Model
import json


class SkeletonPredictor:
    def __init__(self, config_path, model_path, device=None):
        with open(config_path) as f:
            self.config = json.load(f)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.max_frames = checkpoint['max_frames']
        self.embed_dim = checkpoint['embedding_dim']
        graph_args = {
            'layout': checkpoint['layout'],
            'strategy': checkpoint['strategy'],
        }
        self.model = Model(
            in_channels=2,
            embed_dim=self.embed_dim,
            graph_args=graph_args,
            edge_importance_weighting=True
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.centers = checkpoint['center_state_dict']['centers'].to(self.device)
        self.action_map = {i: action['name'] for i, action in enumerate(self.config['actions'])}
        self.num_class = len(self.action_map)
        print('model loaded successfully!\n' +
              'MaxFrames: ' + str(self.max_frames) +
              ' EmbedDim: ' + str(self.embed_dim) +
              ' Layout: ' + str(graph_args['layout']) +
              ' Strategy: ' + str(graph_args['strategy']) + '\n'
                                                            'Val_Metrics:' + str(checkpoint['val_metrics']))

    def _preprocess_window(self, df, start_frame, window_size):
        # 取窗口片段
        x_coords = df.filter(regex='kp_.*_x').values[start_frame:start_frame + window_size]
        y_coords = df.filter(regex='kp_.*_y').values[start_frame:start_frame + window_size]

        # 自动补零处理
        if x_coords.shape[0] < window_size:
            pad_width = ((0, window_size - x_coords.shape[0]), (0, 0))
            x_coords = np.pad(x_coords, pad_width, mode='constant')
            y_coords = np.pad(y_coords, pad_width, mode='constant')

        skeleton_data = np.stack([x_coords, y_coords], axis=0)
        skeleton_data = torch.from_numpy(skeleton_data.astype(np.float32)).unsqueeze(0).unsqueeze(-1)
        return skeleton_data.to(self.device)

    def predict_windows(self, csv_path, step=1, start_mode='first', return_embeddings=False):
        """
        对CSV文件进行滑动窗口预测（支持短序列补零）

        参数:
            csv_path: CSV文件路径
            step: 滑动步长
            start_mode: 'first'（首帧）或'first_nonzero'（第一个非空帧）
            return_embeddings: 是否返回嵌入向量

        返回:
            List[dict]，每个窗口的预测结果
        """
        df = pd.read_csv(csv_path)
        window_size = self.max_frames

        # 判断起始帧（保持原有逻辑）
        if start_mode == 'first_nonzero':
            coords = df.filter(regex='kp_.*_[xy]').values
            nonzero_mask = np.any(coords != 0, axis=1)
            start_idx = np.argmax(nonzero_mask)
        else:
            start_idx = 0

        results = []

        # 计算所有可能的窗口起始位置（包括需要补零的最后一个窗口）
        max_start = max(start_idx, len(df) - window_size) if len(df) >= window_size else start_idx
        window_starts = range(start_idx, max_start + 1, step)

        for start in window_starts:
            # 自动处理短序列补零（在_preprocess_window中实现）
            input_data = self._preprocess_window(df, start, window_size)

            with torch.no_grad():
                embedding = self.model(input_data)
                distances = torch.cdist(embedding, self.centers).squeeze(0)
                min_dist, pred_class = torch.min(distances, 0)
                probabilities = nn.functional.softmax(-distances, dim=0)

            pred_class = pred_class.item()
            min_dist = min_dist.item()
            distances_np = distances.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()

            class_distances = {
                i: {
                    'distance': float(distances_np[i]),
                    'name': self.action_map.get(i, f"Class_{i}")
                } for i in range(self.num_class)
            }

            class_probabilities = {
                i: {
                    'probability': float(probabilities_np[i]),
                    'name': self.action_map.get(i, f"Class_{i}")
                } for i in range(self.num_class)
            }

            result = {
                'window_start': start,
                'window_end': min(start + window_size - 1, len(df) - 1),  # 实际结束帧
                'predicted_class': pred_class,
                'predicted_class_name': self.action_map.get(pred_class, f"Class_{pred_class}"),
                'min_distance': min_dist,
                'class_distances': class_distances,
                'class_probabilities': class_probabilities,
                'is_padded': (start + window_size) > len(df)  # 标记是否补零
            }

            if return_embeddings:
                result['embedding'] = embedding.cpu().numpy()[0]

            results.append(result)

        return results