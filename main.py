#!/usr/bin/env python3
"""
Video2Yolo 主入口
将视频流或视频文件转换为 YOLO 训练数据集
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

import yaml
import cv2

from core.stream_handler import StreamHandler, extract_frames_from_video
from core.annotator import GroundingDINOAnnotator
from core.dataset_builder import DatasetBuilder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Video2Yolo: 视频转 YOLO 数据集")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--source", type=str, help="视频源 (RTSP/RTMP/本地视频)")
    parser.add_argument("--prompt", type=str, help="GroundingDINO 提示词")
    parser.add_argument("--fps", type=int, help="抽帧帧率")
    parser.add_argument("--output", type=str, help="输出目录")
    parser.add_argument("--device", type=str, help="设备 (cuda/cpu)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def process_stream(args, config):
    """处理视频流"""
    source = args.source or config.get("source")
    fps = args.fps or config.get("fps", 10)
    output_dir = Path(args.output or config.get("output_dir", "outputs"))
    prompt = args.prompt or config.get("prompt", "electric meter")

    # 创建临时目录存放帧和标签
    temp_dir = tempfile.mkdtemp(prefix="video2yolo_")
    temp_path = Path(temp_dir)
    frames_dir = temp_path / "frames"
    labels_dir = temp_path / "labels"
    frames_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    logger.info(f"临时目录: {temp_dir}")

    # 初始化标注器
    annotator = GroundingDINOAnnotator(
        config_path=config["model_config"],
        checkpoint_path=config["model_checkpoint"],
        prompt=prompt,
        box_threshold=config.get("box_threshold", 0.35),
        text_threshold=config.get("text_threshold", 0.25),
        device=args.device or config.get("device", "cuda")
    )
    annotator.load_model()

    # 处理视频源
    logger.info(f"开始处理视频源: {source}")

    # 判断是本地视频还是流媒体
    is_stream = source.startswith(("rtsp://", "rtmp://", "http://"))

    if is_stream:
        # 流媒体模式：边拉边标注
        handler = StreamHandler(source, fps=fps, output_dir=frames_dir)

        if not handler.open():
            logger.error("无法打开视频流")
            return

        frame_idx = 0
        handler.start_capture()

        try:
            while True:
                frame = handler.read_frame()
                if frame is None:
                    continue

                # 标注当前帧
                results = annotator.annotate_frame(frame)

                # 保存图片
                img_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(img_path), frame)

                # 保存标签
                label_path = labels_dir / f"frame_{frame_idx:06d}.txt"
                if results:
                    h, w = frame.shape[:2]
                    with open(label_path, "w") as f:
                        for r in results:
                            cx, cy, bw, bh = r["bbox"]
                            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                frame_idx += 1

                if frame_idx % 100 == 0:
                    logger.info(f"已处理 {frame_idx} 帧")

                # 可选：限制最大帧数
                # if frame_idx >= 1000:
                #     break

        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            handler.stop_capture()
            handler.close()

    else:
        # 本地视频模式：先提取所有帧，再批量标注
        logger.info("本地视频模式")
        frames = extract_frames_from_video(source, fps=fps, output_dir=frames_dir)

        # 批量标注
        stats = annotator.batch_annotate(frames_dir, labels_dir)
        logger.info(f"标注完成: {stats}")

    # 构建数据集
    logger.info("构建 YOLO 数据集...")

    builder = DatasetBuilder(
        output_dir=output_dir,
        class_names=config.get("class_names", ["electric_meter"]),
        split_ratio=config.get("split_ratio", 0.8)
    )

    dataset_dir = builder.build_from_directories(
        images_dir=frames_dir,
        labels_dir=labels_dir,
        copy=True
    )

    logger.info(f"数据集生成完成: {dataset_dir}")

    # 清理临时目录
    shutil.rmtree(temp_dir)
    logger.info("临时文件已清理")

    # 输出 data.yaml 路径
    print(f"\n数据集路径: {dataset_dir}")
    print(f"训练命令: yolo task=detect mode=train data={dataset_dir}/data.yaml model=yolov8n.pt epochs=100")


def main():
    args = parse_args()

    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        # 默认配置
        config = {
            "source": args.source or "test.mp4",
            "prompt": args.prompt or "electric meter",
            "fps": args.fps or 10,
            "output_dir": args.output or "outputs",
            "box_threshold": 0.35,
            "text_threshold": 0.25,
            "split_ratio": 0.8,
            "class_names": ["electric_meter"],
            "device": args.device or "cuda",
            "model_config": "/home/stephen/.openclaw/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "model_checkpoint": "/home/stephen/.openclaw/workspace/models/AI-ModelScope/GroundingDINO/groundingdino_swint_ogc.pth"
        }

    # 检查模型文件是否存在
    model_checkpoint = config.get("model_checkpoint")
    if model_checkpoint and not Path(model_checkpoint).exists():
        logger.warning(f"模型文件不存在: {model_checkpoint}")
        logger.info("请下载模型权重或修改配置文件中的 model_checkpoint 路径")

    process_stream(args, config)


if __name__ == "__main__":
    main()
