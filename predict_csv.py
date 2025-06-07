import argparse

from predictor import SkeletonPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='骨骼动作识别滑动窗口预测')
    parser.add_argument('--csv_file', type=str, default=r'D:\Desktop\Study\LbSafe\yolo-pose\output.csv', help='输入的CSV文件路径')
    parser.add_argument('--config', type=str, default=r'config.json', help='配置文件路径')
    parser.add_argument('--model', type=str, default=r'./checkpoints/yolopose/spatial/best_model.pth', help='模型权重路径')
    parser.add_argument('--step', type=int, default=10, help='滑动步长')
    parser.add_argument('--start_mode', type=str, default='first_nonzero', choices=['first', 'first_nonzero'], help='窗口起始方式')
    args = parser.parse_args()

    predictor = SkeletonPredictor(args.config, args.model)
    results = predictor.predict_windows(
        args.csv_file,
        step=args.step,
        start_mode=args.start_mode
    )
    for i, res in enumerate(results):
        print(f"\n窗口{i+1}: 帧{res['window_start']}~{res['window_end']}")
        print(f"预测类别ID: {res['predicted_class']}")
        print(f"预测类别名称: {res['predicted_class_name']}")
        print(f"与最近中心的距离: {res['min_distance']:.4f}")
        print("各类别距离:", {k: v['distance'] for k, v in res['class_distances'].items()})
        print("各类别概率:", {k: f"{v['probability']:.2%}" for k, v in res['class_probabilities'].items()})