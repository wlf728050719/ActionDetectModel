import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from predictor import SkeletonPredictor
import json
from typing import Optional, Dict, Tuple
import argparse

class EmbeddingVisualizer:
    def __init__(self, config_path: str, model_path: str, device: Optional[str] = None):
        """
        Initialize the visualizer with model and config.

        Args:
            config_path: Path to config JSON file.
            model_path: Path to trained model checkpoint.
            device: Device to use (cuda/cpu), auto-detected if None.
        """
        self.predictor = SkeletonPredictor(config_path, model_path, device)
        with open(config_path) as f:
            self.config = json.load(f)
        self.label_names = [action['name'] for action in self.config['actions']]

    def load_embeddings(
            self,
            mode: str = 'single',
            window_config: Optional[Dict] = None,
            max_samples_per_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load embeddings from all CSV files.

        Args:
            mode: 'single' (first window only) or 'sliding' (multiple windows per CSV).
            window_config: Configuration for sliding window (required for 'sliding' mode).
                          Should contain {'step': int, 'start_mode': str}.
            max_samples_per_class: Maximum number of samples to load per class.

        Returns:
            Tuple of (embeddings, labels) as numpy arrays.
        """
        if mode == 'sliding' and window_config is None:
            raise ValueError("window_config must be provided for sliding mode")

        all_embeddings = []
        all_labels = []

        for action in self.config['actions']:
            data_folder = action['data_folder']
            label = action['label']
            samples_loaded = 0

            print(f"Processing {action['name']} files...")

            for csv_file in os.listdir(data_folder):
                if csv_file.endswith('.csv'):
                    csv_path = os.path.join(data_folder, csv_file)

                    try:
                        if mode == 'single':
                            # Get only the first window
                            result = self.predictor.predict_windows(
                                csv_path,
                                step=self.predictor.max_frames,  # Ensure single window
                                start_mode='first',
                                return_embeddings=True
                            )
                            if result:
                                all_embeddings.append(result[0]['embedding'])
                                all_labels.append(label)
                                samples_loaded += 1

                        elif mode == 'sliding':
                            # Get multiple windows per CSV
                            results = self.predictor.predict_windows(
                                csv_path,
                                step=window_config['step'],
                                start_mode=window_config['start_mode'],
                                return_embeddings=True
                            )
                            for res in results:
                                all_embeddings.append(res['embedding'])
                                all_labels.append(label)
                                samples_loaded += 1

                        if max_samples_per_class is not None and 0 < max_samples_per_class <= samples_loaded:
                            break

                    except Exception as e:
                        print(f"Error processing {csv_path}: {str(e)}")

        return np.array(all_embeddings), np.array(all_labels)

    def visualize_pca(
            self,
            output_path: str = 'visualizations/pca_with_centers.png',
            mode: str = 'single',
            window_config: Optional[Dict] = None,
            max_samples_per_class: Optional[int] = None,
            figsize: Tuple[int, int] = (12, 10),
            dpi: int = 300
    ):
        """
        Generate and save PCA visualization.

        Args:
            output_path: Path to save the visualization.
            mode: 'single' or 'sliding' window mode.
            window_config: Configuration for sliding window.
            max_samples_per_class: Maximum samples per class to visualize.
            figsize: Figure size.
            dpi: Image DPI.
        """
        # Load embeddings
        embeddings, labels = self.load_embeddings(
            mode=mode,
            window_config=window_config,
            max_samples_per_class=max_samples_per_class
        )

        # Get centers
        centers = self.predictor.centers.cpu().numpy()

        # Combine embeddings and centers for PCA
        all_points = np.vstack([embeddings, centers])
        extended_labels = np.concatenate([labels, np.arange(len(centers))])

        # Perform PCA
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(all_points)

        # Separate data points and centers
        data_points_2d = points_2d[:-len(centers)]
        centers_2d = points_2d[-len(centers):]

        # Create plot
        plt.figure(figsize=figsize)
        colors = plt.cm.get_cmap('tab10', len(self.label_names))

        # Plot data points
        for label in np.unique(labels):
            mask = labels == label
            plt.scatter(
                data_points_2d[mask, 0], data_points_2d[mask, 1],
                color=colors(label),
                label=self.label_names[label],
                alpha=0.6,
                s=50
            )

        # Plot centers
        for i, (center_x, center_y) in enumerate(centers_2d):
            plt.scatter(
                center_x, center_y,
                color=colors(i),
                marker='*',
                s=400,
                edgecolor='black',
                linewidth=1,
                label=f'{self.label_names[i]} Center'
            )

        # Configure plot
        title = 'PCA Visualization of Action Embeddings'
        if mode == 'sliding':
            title += f" (Sliding Window, step={window_config['step']})"
        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"PCA visualization saved to {output_path}")
        print(f"Number of data points in this plot: {len(embeddings)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize action embeddings using PCA.")
    parser.add_argument('--config', type=str, default='config.json', help='Path to config JSON file.')
    parser.add_argument('--model', type=str, default='checkpoints/yolopose/spatial/best_model.pth',
                        help='Path to trained model checkpoint.')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Directory to save visualizations.')
    parser.add_argument('--max_samples_single', type=int, default=-1,
                        help='Max samples per class for single window mode. Negative means all samples.')
    parser.add_argument('--max_samples_sliding', type=int, default=-1,
                        help='Max samples per class for sliding window mode. Negative means all samples.')
    parser.add_argument('--sliding_step', type=int, default=10, help='Step size for sliding window.')
    parser.add_argument('--sliding_start_mode', type=str, default='first_nonzero',
                        help='Start mode for sliding window.')
    args = parser.parse_args()

    # Initialize visualizer
    visualizer = EmbeddingVisualizer(args.config, args.model)

    # Mode 1: Single window per CSV (first max_frames)
    output_path_single = os.path.join(args.output_dir, 'pca_single_window.png')
    visualizer.visualize_pca(
        output_path=output_path_single,
        mode='single',
        max_samples_per_class=args.max_samples_single
    )

    # Mode 2: Sliding window
    window_config = {
        'step': args.sliding_step,
        'start_mode': args.sliding_start_mode
    }
    output_path_sliding = os.path.join(args.output_dir, f'pca_sliding_window_step{args.sliding_step}.png')
    visualizer.visualize_pca(
        output_path=output_path_sliding,
        mode='sliding',
        window_config=window_config,
        max_samples_per_class=args.max_samples_sliding
    )
