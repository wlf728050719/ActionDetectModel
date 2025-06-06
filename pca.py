import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from predict import SkeletonPredictor
import json


def load_all_csv_embeddings(config_path, predictor):
    """加载所有行为类别的CSV文件并提取嵌入向量"""
    with open(config_path) as f:
        config = json.load(f)

    all_embeddings = []
    all_labels = []
    label_names = []

    for action in config['actions']:
        data_folder = action['data_folder']
        label = action['label']
        label_name = action['name']
        label_names.append(label_name)

        print(f"Processing {label_name} files...")

        # 遍历该行为类别的所有CSV文件
        for csv_file in os.listdir(data_folder):
            if csv_file.endswith('.csv'):
                csv_path = os.path.join(data_folder, csv_file)
                try:
                    result = predictor.predict(csv_path)
                    embedding = result['embedding']
                    all_embeddings.append(embedding)
                    all_labels.append(label)
                except Exception as e:
                    print(f"Error processing {csv_path}: {str(e)}")

    return np.array(all_embeddings), np.array(all_labels), label_names


def plot_pca_with_centers(embeddings, labels, centers, label_names):
    """执行PCA降维并绘制结果"""
    # 合并嵌入向量和中心点
    all_points = np.vstack([embeddings, centers])
    extended_labels = np.concatenate([labels, np.arange(len(centers))])

    # PCA降维到2D
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(all_points)

    # 分离数据点和中心点
    data_points_2d = points_2d[:-len(centers)]
    centers_2d = points_2d[-len(centers):]

    # 创建颜色映射
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    plt.figure(figsize=(12, 10))

    # 绘制数据点
    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            data_points_2d[mask, 0], data_points_2d[mask, 1],
            color=colors(label),
            label=label_names[label],
            alpha=0.6,
            s=50
        )

    # 绘制中心点（更大、更醒目的标记）
    for i, (center_x, center_y) in enumerate(centers_2d):
        plt.scatter(
            center_x, center_y,
            color=colors(i),
            marker='*',
            s=400,
            edgecolor='black',
            linewidth=1,
            label=f'{label_names[i]} Center'
        )

    plt.title('PCA Visualization of Action Embeddings with Class Centers')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/pca_with_centers.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    config_path = 'config.json'
    model_path = 'checkpoints/yolopose/spatial/best_model.pth'
    layout = 'yolopose'
    strategy = 'spatial'
    embed_dim = 4096
    max_frames = 50

    predictor = SkeletonPredictor(
        config_path=config_path,
        model_path=model_path,
        layout=layout,
        strategy=strategy,
        embed_dim=embed_dim,
        max_frames=max_frames
    )

    # 加载所有CSV文件并提取嵌入向量
    embeddings, labels, label_names = load_all_csv_embeddings(config_path, predictor)

    # 获取中心点（转换为numpy数组）
    centers = predictor.centers.cpu().numpy()

    # 绘制PCA结果
    plot_pca_with_centers(embeddings, labels, centers, label_names)


if __name__ == "__main__":
    main()