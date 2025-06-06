import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split


class SkeletonDataset(Dataset):
    def __init__(self, data_paths, labels, max_frames=50):
        """
        骨骼动作识别数据集

        参数:
            data_paths: CSV文件路径列表
            labels: 对应的标签列表
            max_frames: 每个样本的最大帧数 (不足的补零)
        """
        self.data_paths = data_paths
        self.labels = labels
        self.max_frames = max_frames

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        csv_path = self.data_paths[idx]
        label = self.labels[idx]

        # 读取CSV文件
        df = pd.read_csv(csv_path)
        num_frames = min(len(df), self.max_frames)

        # 提取x,y坐标 (忽略置信度)
        x_coords = df.filter(regex='kp_.*_x').values[:self.max_frames]
        y_coords = df.filter(regex='kp_.*_y').values[:self.max_frames]

        # 组合成形状为 [2, T, V] 的张量 (T=时间步, V=关键点)
        skeleton_data = np.stack([x_coords, y_coords], axis=0)  # [2, T, V]

        # 如果帧数不足max_frames，则补零
        if skeleton_data.shape[1] < self.max_frames:
            pad_width = ((0, 0), (0, self.max_frames - skeleton_data.shape[1]), (0, 0))
            skeleton_data = np.pad(skeleton_data, pad_width, mode='constant')

        # 转换为float32并添加通道维度 (适配模型输入形状)
        skeleton_data = torch.from_numpy(skeleton_data.astype(np.float32)).unsqueeze(-1)

        return skeleton_data, label


def get_dataloader(config_path, batch_size=32, shuffle=True, num_workers=4, test_size=0.2, random_state=42):
    """
    获取训练和验证数据加载器

    参数:
        config_path: config.json路径
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        test_size: 验证集比例
        random_state: 随机种子
    """
    with open(config_path) as f:
        config = json.load(f)

    data_paths = []
    labels = []

    # 收集所有CSV文件和对应标签
    for action in config['actions']:
        label = int(action['label'])
        data_folder = action['data_folder']

        if not os.path.exists(data_folder):
            continue

        for csv_file in os.listdir(data_folder):
            if csv_file.endswith('.csv'):
                data_paths.append(os.path.join(data_folder, csv_file))
                labels.append(label)

    # 统计类别分布
    class_counts = np.bincount(labels)
    print(f"数据集统计: 总样本数={len(data_paths)}, 类别分布={class_counts}")

    # 划分训练集和验证集 (保持类别分布)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        data_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # 创建数据集
    train_dataset = SkeletonDataset(train_paths, train_labels)
    val_dataset = SkeletonDataset(val_paths, val_labels)

    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 丢弃最后一个不完整的batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要shuffle
        num_workers=num_workers,
        pin_memory=True
    )

    return len(class_counts),train_loader, val_loader


if __name__ == "__main__":
    _,train_loader, val_loader = get_dataloader('config.json')

    # 加载一组训练数据
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"\n训练集 Batch {batch_idx}:")
        print("数据形状:", data.shape)  # 应该是 [batch_size, 2, 50, 17, 1]
        print("标签:", labels.numpy())
        break

    # 加载一组验证数据
    for batch_idx, (data, labels) in enumerate(val_loader):
        print(f"\n验证集 Batch {batch_idx}:")
        print("数据形状:", data.shape)
        print("标签:", labels.numpy())
        break
