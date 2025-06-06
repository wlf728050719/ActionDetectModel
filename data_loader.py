import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random


def triplet_collate_fn(batch):
    """自定义collate函数处理三元组"""
    anchors = torch.stack([item[0][0] for item in batch])
    positives = torch.stack([item[0][1] for item in batch])
    negatives = torch.stack([item[0][2] for item in batch])
    anchor_labels = torch.tensor([item[1][0] for item in batch])
    return (anchors, positives, negatives), anchor_labels


def regular_collate_fn(batch):
    """常规collate函数处理数据和标签"""
    data = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return data, labels


class SkeletonDataset(Dataset):
    """基础骨骼数据集，负责数据加载"""

    def __init__(self, data_paths, labels, max_frames=50):
        """
        骨骼动作识别基础数据集

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
        data = self._load_single_sample(self.data_paths[idx])
        label = self.labels[idx]
        # 判断label是否与path对应
        # print(label)
        # print(self.data_paths[idx])
        return data, label

    def _load_single_sample(self, csv_path):
        """加载单个样本的骨骼数据"""
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

        return skeleton_data


class TripletSkeletonDataset(Dataset):
    """三元组骨骼数据集，基于基础数据集生成三元组"""

    def __init__(self, base_dataset):
        """
        骨骼动作识别三元组数据集

        参数:
            base_dataset: 基础骨骼数据集实例
        """
        self.base_dataset = base_dataset
        self.labels = base_dataset.labels

        # 创建标签到样本索引的映射
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        # 获取所有类别列表
        self.labels_set = set(self.labels)
        self.labels_list = list(self.labels_set)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 首先获取anchor样本
        anchor_data, anchor_label = self.base_dataset[idx]

        # 随机选择一个正样本(相同类别)
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = random.choice(self.label_to_indices[anchor_label])

        # 随机选择一个负样本(不同类别)
        negative_label = random.choice(list(self.labels_set - {anchor_label}))
        negative_idx = random.choice(self.label_to_indices[negative_label])

        # 加载正负样本的数据
        positive_data, _ = self.base_dataset[positive_idx]
        negative_data, _ = self.base_dataset[negative_idx]

        return (anchor_data, positive_data, negative_data), [anchor_label]


def get_dataloader(config_path, batch_size=32, shuffle=True, num_workers=4, test_size=0.2, random_state=42):
    """
    获取三元组训练和验证数据加载器

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
        label = action['label']  # 直接使用config中的label字段
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

    # 创建基础数据集
    train_base_dataset = SkeletonDataset(train_paths, train_labels)
    val_base_dataset = SkeletonDataset(val_paths, val_labels)

    # 创建三元组训练数据集
    train_dataset = TripletSkeletonDataset(train_base_dataset)
    val_dataset = val_base_dataset  # 验证集使用常规数据集

    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=triplet_collate_fn,  # 训练使用三元组collate函数
        drop_last=True  # 丢弃最后一个不完整的batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要shuffle
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=regular_collate_fn  # 验证使用常规collate函数
    )

    return len(class_counts), train_loader, val_loader


if __name__ == "__main__":
    num_classes, train_loader, val_loader = get_dataloader('config.json')

    # 加载一组训练数据
    for batch_idx, ((anchors, positives, negatives), labels) in enumerate(train_loader):
        print(f"\n训练集 Batch {batch_idx}:")
        print("Anchor形状:", anchors.shape)  # [batch_size, 2, 50, 17, 1]
        print("Positive形状:", positives.shape)  # [batch_size, 2, 50, 17, 1]
        print("Negative形状:", negatives.shape)
        print(labels)# [batch_size, 2, 50, 17, 1]
        break

    # 加载一组验证数据
    for batch_idx, (data, labels) in enumerate(val_loader):
        print(f"\n验证集 Batch {batch_idx}:")
        print("数据形状:", data.shape)  # [batch_size, 2, 50, 17, 1]
        print("标签形状:", labels.shape)  # [batch_size]
        print("标签示例:", labels[:5])  # 打印前5个标签
        break