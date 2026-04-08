# Video2Yolo

将视频流或视频文件转换为 YOLO 训练数据集的自动化工具。

## 功能

- 输入：RTSP/RTMP 拉流地址 或 本地视频文件
- 输出：可直接用于 YOLO 训练的 dataset
- 工作流程：
  1. 拉流/读取视频
  2. 视频拆帧（10 FPS）
  3. GroundingDINO 自动标注（标签: electric_meter）
  4. 8:2 拆分训练集/验证集

## 目录结构

```
Video2Yolo/
├── core/
│   ├── stream_handler.py    # 拉流与抽帧
│   ├── annotator.py         # GroundingDINO 标注
│   └── dataset_builder.py   # 数据集构建与拆分
├── config/
│   └── config.yaml          # 配置文件
├── weights/                 # GroundingDINO 权重
├── outputs/                 # 输出数据集
├── main.py                  # 主入口
└── README.md
```

## 安装

```bash
# 1. 克隆本项目
git clone https://github.com/chenwen0511/Video2Yolo.git
cd Video2Yolo

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 下载并安装 GroundingDINO
cd /home/nvidia/stephen  # 或任意目录
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .

# 4. 下载模型权重
mkdir -p weights
wget -O weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

> 注意：代码会自动检测 GroundingDINO 路径（依次查找 `/home/nvidia/stephen/GroundingDINO`、`/home/stephen/.openclaw/workspace/GroundingDINO`、`~/GroundingDINO`），如需自定义请修改 `core/annotator.py`

## 配置

编辑 `config/config.yaml`:

```yaml
source: "rtsp://admin:12345@192.168.1.100/stream"  # 拉流地址或视频文件路径
prompt: "electric meter"
fps: 10
box_threshold: 0.35
text_threshold: 0.25
split_ratio: 0.8
output_dir: "outputs"
```

## 使用

```bash
# 使用配置文件
python main.py --config config/config.yaml

# 或命令行参数
python main.py --source rtsp://admin:12345@192.168.1.100/stream --prompt "electric meter"
```

## 输出

```
outputs/dataset_YYYYMMDD/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

## 快速开始

```bash
# 1. 配置 config.yaml
# 2. 运行
python main.py --config config/config.yaml

# 3. 用生成的数据集训练 YOLO
yolo task=detect mode=train data=outputs/dataset_xxx/data.yaml model=yolov8n.pt epochs=100
```
