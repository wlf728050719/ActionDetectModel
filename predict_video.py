from get_pose import process_video
from predictor import SkeletonPredictor

def predict_video(
        video_path,
        pose_model_path,
        action_model_path,
        action_config_path,
        output_csv_path=None,
        output_video_path=None,
        conf_threshold=0.1,
        step=1,
        start_mode='first',
        return_embeddings=False,
        device=None
):
    """
    处理视频并预测动作类别

    参数:
        video_path: 输入视频路径
        pose_model_path: 姿态估计模型路径
        action_model_path: 动作分类模型路径
        action_config_path: 动作配置文件路径
        output_csv_path: (可选)输出骨骼数据CSV路径，如果为None则不保存
        output_video_path: (可选)输出可视化视频路径，如果为None则不保存
        conf_threshold: 姿态估计置信度阈值
        step: 滑动窗口步长
        start_mode: 'first'或'first_nonzero'，窗口起始模式
        return_embeddings: 是否返回嵌入向量
        device: 使用的设备(CPU/GPU)

    返回:
        预测结果列表(List[dict])
    """
    import tempfile
    import os

    # 如果未提供输出CSV路径，创建临时文件
    temp_csv = None
    if output_csv_path is None:
        temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        output_csv_path = temp_csv.name
        temp_csv.close()

    # 1. 处理视频生成骨骼数据
    print("正在提取视频骨骼数据...")
    process_video(
        model_path=pose_model_path,
        input_video_path=video_path,
        output_csv_path=output_csv_path,
        output_video_path=output_video_path,
        conf_threshold=conf_threshold
    )

    # 2. 初始化预测器
    print("初始化动作分类器...")
    predictor = SkeletonPredictor(
        config_path=action_config_path,
        model_path=action_model_path,
        device=device
    )

    # 3. 进行预测
    print("正在进行动作分类预测...")
    results = predictor.predict_windows(
        csv_path=output_csv_path,
        step=step,
        start_mode=start_mode,
        return_embeddings=return_embeddings
    )
    # 清理临时文件
    if temp_csv is not None:
        os.unlink(output_csv_path)

    print("视频预测完成!")
    return results

if __name__ == "__main__":
    # 示例调用
    results = predict_video(
        video_path=r"D:\Desktop\0pQACKjlllc.mp4",
        pose_model_path="yolov5s6_pose.onnx",
        action_model_path="checkpoints/yolopose/spatial/best_model.pth",
        action_config_path="config.json",
        output_csv_path=None,
        output_video_path='output.mp4',
        conf_threshold=0.3,
        step=10,
        start_mode='first_nonzero'
    )

    # 打印结果
    for result in results:
        print(f"窗口 {result['window_start']}-{result['window_end']}: "
              f"预测动作: {result['predicted_class_name']} "
              f"(置信度: {result['class_probabilities'][result['predicted_class']]['probability']:.2f})"
              f"距离: {result['min_distance']} ")