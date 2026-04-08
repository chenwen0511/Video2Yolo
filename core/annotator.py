#!/usr/bin/env python3
"""GroundingDINO 自动标注模块"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GroundingDINOAnnotator:
    """GroundingDINO 零样本标注器"""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        prompt: str = "electric meter",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cuda"
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.model = None

        # 类别映射
        self.class_names = ["electric_meter"]

    def load_model(self):
        """加载 GroundingDINO 模型"""
        logger.info("加载 GroundingDINO 模型...")

        from groundingdino.util.inference import load_model

        self.model = load_model(
            self.config_path,
            self.checkpoint_path,
            device=self.device
        )

        logger.info(f"模型加载成功，设备: {self.device}")

    def _load_image(self, image_path: str):
        """加载图片"""
        from groundingdino.util.inference import load_image

        image_source, image_transformed = load_image(image_path)
        return image_source, image_transformed

    def predict(self, image_source, image_transformed):
        """预测边界框"""
        from groundingdino.util.inference import predict

        boxes, logits, phrases = predict(
            self.model,
            image_transformed,
            self.prompt,
            self.box_threshold,
            self.text_threshold,
            device=self.device
        )

        return boxes, logits, phrases

    def predict_from_image(self, image_source):
        """从已加载的图片预测"""
        from groundingdino.util.inference import predict

        boxes, logits, phrases = predict(
            self.model,
            image_source,
            self.prompt,
            self.box_threshold,
            self.text_threshold,
            device=self.device
        )

        return boxes, logits, phrases

    def annotate_frame(self, frame) -> list[dict]:
        """
        对单帧进行标注

        Args:
            frame: numpy array (H, W, C) BGR 格式

        Returns:
            检测结果列表 [{"class_id": 0, "bbox": [cx, cy, w, h], "conf": 0.xx}, ...]
        """
        # 转换 BGR -> RGB
        image_rgb = frame[:, :, ::-1]
        image_pil = Image.fromarray(image_rgb)

        # 转换为 GroundingDINO 需要的格式
        import groundingdino.util.inference as inference
        image_transformed = inference.transform(image_pil).to(self.device)

        boxes, logits, phrases = self.predict(image_transformed, image_transformed)

        results = []
        h, w = frame.shape[:2]

        if len(boxes) > 0:
            for box, logit, phrase in zip(boxes, logits, phrases):
                # 归一化的 cx, cy, w, h
                cx, cy, bw, bh = box.tolist()
                results.append({
                    "class_id": 0,
                    "bbox": [cx, cy, bw, bh],
                    "conf": float(logit),
                    "phrase": phrase
                })

        return results

    def annotate_image_file(
        self,
        image_path: str,
        output_label_path: Optional[str] = None
    ) -> list[dict]:
        """
        对图片文件进行标注

        Args:
            image_path: 输入图片路径
            output_label_path: 输出标签文件路径（YOLO格式）

        Returns:
            检测结果列表
        """
        from groundingdino.util.inference import load_image

        image_source, image_transformed = load_image(str(image_path))
        h, w = image_source.shape[:2]

        boxes, logits, phrases = self.predict(image_source, image_transformed)

        results = []
        if len(boxes) > 0:
            for box, logit, phrase in zip(boxes, logits, phrases):
                cx, cy, bw, bh = box.tolist()
                results.append({
                    "class_id": 0,
                    "bbox": [cx, cy, bw, bh],
                    "conf": float(logit),
                    "phrase": phrase
                })

        # 保存 YOLO 格式标签
        if output_label_path and results:
            self._save_yolo_label(output_label_path, results, w, h)

        return results

    def _save_yolo_label(self, label_path: str, results: list[dict], w: int, h: int):
        """保存 YOLO 格式标签"""
        with open(label_path, "w") as f:
            for r in results:
                cx, cy, bw, bh = r["bbox"]
                conf = r["conf"]
                class_id = r["class_id"]
                # YOLO 格式: class_id cx cy w h
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        logger.debug(f"标签已保存: {label_path}")

    def batch_annotate(
        self,
        image_dir: Path,
        output_dir: Path,
        exts: tuple = (".jpg", ".jpeg", ".png")
    ) -> dict:
        """
        批量标注图片

        Args:
            image_dir: 图片目录
            output_dir: 输出标签目录
            exts: 支持的图片扩展名

        Returns:
            统计信息
        """
        image_files = sorted(image_dir.glob(f"*{exts[0]}"))
        for ext in exts[1:]:
            image_files.extend(sorted(image_dir.glob(f"*{ext}")))

        logger.info(f"找到 {len(image_files)} 张图片")

        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "total": len(image_files),
            "annotated": 0,
            "empty": 0,
            "total_objects": 0
        }

        for img_path in tqdm(image_files, desc="标注中"):
            label_path = output_dir / f"{img_path.stem}.txt"

            results = self.annotate_image_file(str(img_path), str(label_path))

            if results:
                stats["annotated"] += 1
                stats["total_objects"] += len(results)
                logger.debug(f"{img_path.name}: 检测到 {len(results)} 个目标")
            else:
                stats["empty"] += 1

        logger.info(f"标注完成: {stats['annotated']}/{stats['total']} 张有目标, "
                   f"共 {stats['total_objects']} 个对象")

        return stats


if __name__ == "__main__":
    # 测试
    import yaml

    logging.basicConfig(level=logging.INFO)

    # 加载配置
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    annotator = GroundingDINOAnnotator(
        config_path=config["model_config"],
        checkpoint_path=config["model_checkpoint"],
        prompt=config["prompt"],
        box_threshold=config["box_threshold"],
        text_threshold=config["text_threshold"],
        device=config.get("device", "cuda")
    )

    annotator.load_model()

    # 测试单图标注
    # results = annotator.annotate_image_file("test.jpg", "test.txt")
    # print(results)
