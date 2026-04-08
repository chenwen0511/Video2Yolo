#!/usr/bin/env python3
"""视频流拉取与抽帧模块"""

import cv2
import time
import threading
from pathlib import Path
from queue import Queue
from typing import Optional, Generator
import logging

logger = logging.getLogger(__name__)


class StreamHandler:
    """视频流处理器，支持拉流地址和本地视频文件"""

    def __init__(
        self,
        source: str,
        fps: int = 10,
        output_dir: Optional[Path] = None
    ):
        self.source = source
        self.target_fps = fps
        self.frame_interval = 1.0 / fps
        self.output_dir = output_dir
        self.cap = None
        self.running = False
        self.frame_queue: Queue = Queue(maxsize=30)
        self.thread: Optional[threading.Thread] = None

        # 状态
        self.total_frames = 0
        self.dropped_frames = 0

    def open(self) -> bool:
        """打开视频源"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"无法打开视频源: {self.source}")
            return False
        logger.info(f"视频源已打开: {self.source}")
        return True

    def close(self):
        """关闭视频源"""
        if self.cap:
            self.cap.release()
            logger.info("视频源已关闭")

    def _capture_loop(self):
        """后台捕获线程"""
        last_time = time.time()

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                # 尝试重连（如果是网络流）
                logger.warning("视频流断开，尝试重连...")
                self.cap.release()
                time.sleep(1)
                if not self.cap.open(self.source):
                    logger.error("重连失败")
                    break
                continue

            self.total_frames += 1

            # 控制帧率
            elapsed = time.time() - last_time
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)

            last_time = time.time()

            # 放入队列（阻塞等待消费者）
            self.frame_queue.put(frame.copy())

        logger.info("捕获线程结束")

    def start_capture(self):
        """启动后台捕获"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("后台捕获已启动")

    def stop_capture(self):
        """停止捕获"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("后台捕获已停止")

    def read_frame(self) -> Optional:
        """读取一帧（非阻塞）"""
        return self.frame_queue.get() if not self.frame_queue.empty() else None

    def get_frame_generator(self) -> Generator:
        """获取帧生成器（同步模式）"""
        last_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 控制帧率
            elapsed = time.time() - last_time
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)
            last_time = time.time()

            yield frame

    def save_frame(self, frame, filename: str) -> Path:
        """保存帧到文件"""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / filename
            cv2.imwrite(str(filepath), frame)
            return filepath
        return None


def extract_frames_from_video(
    video_path: str,
    fps: int = 10,
    output_dir: Optional[Path] = None,
    max_frames: Optional[int] = None
) -> list[Path]:
    """
    从本地视频文件提取帧

    Args:
        video_path: 视频文件路径
        fps: 提取帧率
        output_dir: 输出目录
        max_frames: 最大帧数限制

    Returns:
        保存的帧文件路径列表
    """
    handler = StreamHandler(video_path, fps=fps, output_dir=output_dir)

    if not handler.open():
        return []

    saved_frames = []
    frame_idx = 0

    for frame in handler.get_frame_generator():
        if max_frames and frame_idx >= max_frames:
            break

        filename = f"frame_{frame_idx:06d}.jpg"
        filepath = handler.save_frame(frame, filename)
        if filepath:
            saved_frames.append(filepath)

        frame_idx += 1

    handler.close()
    logger.info(f"共提取 {len(saved_frames)} 帧")

    return saved_frames


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)

    # 测试本地视频
    # frames = extract_frames_from_video("test.mp4", fps=10, output_dir=Path("temp_frames"))
    # print(f"提取了 {len(frames)} 帧")
