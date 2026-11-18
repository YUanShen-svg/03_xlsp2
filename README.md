# 实拍人物 + 头部检测 —— YOLO 双类别模型训练

第四节课作业 · YOLO 目标检测与神经网络基础

## 1. 视频来源

| 项 | 内容 |
|---|---|
| 视频来源 | B 站实拍人物视频（自选片段） |
| 原始属性 | 1080×1920（竖屏）/ 30 fps / 372 帧 / 12.4 秒 |
| 抽帧方式 | 每 0.25 秒抽一帧，共抽出 46 张 |
| 场景说明 | 竖屏实拍，画面中有行人及其头部 |

## 2. 类别定义

| 类别序号 | 类别名 | 说明 |
|---|---|---|
| 0 | `person` | 画面中的行人（COCO 已有类别） |
| 1 | `head` | 行人的头部（COCO 无此类别，由关键点模型反推） |

**类别数量：nc = 2**

与数据集配置文件 `xlsp2.yaml` 中的 `nc: 2` 和 `names: ['person', 'head']` 完全一致。

### 关于 `head` 类别的由来

COCO 预训练模型没有 `head` 这个类别，无法直接预标注。这里使用 **yolo26s-pose 关键点模型**先检测每个人的 5 个面部关键点（鼻、双眼、双耳），再由关键点外接框反推出头部区域，转换成 YOLO 检测格式的框。这样第二类不用手工一个个画。

## 3. 数据集统计

| 划分 | 图片数 | person 标注数 | head 标注数 | 合计框数 |
|---|---|---|---|---|
| train | 37 | 37 | 37 | 74 |
| val | 9 | 9 | 9 | 18 |
| **合计** | **46** | **46** | **46** | **92** |

- train / val 划分比例：**8 : 2**（37 : 9）
- **train 与 val 无重叠**：已校验文件名与内容，确认无重叠
- 每个行人都同时有 `person` 和 `head` 两个框，两类数量严格相等
- 无空标签图片（46 张图全部有标注）

### 标注流程

1. **`person` 类**：用 COCO 预训练 YOLO26s 自动预标注
2. **`head` 类**：用 yolo26s-pose 检测面部关键点，由关键点外接框反推头部位置
3. **统一转换**：两类合并进同一份 YOLO 格式 `.txt`，类别序号 0 = person、1 = head
4. **一致性校验**：校验每张图中 person 与 head 数量相等，确保没有漏标

## 4. 数据集目录结构

```
datasets/xlsp2/
├── images/
│   ├── train/      # 37 张训练图片
│   └── val/        # 9 张验证图片
├── labels/
│   ├── train/      # 与 train 图片同名的 .txt
│   └── val/
└── xlsp2.yaml
```

标注文件为 YOLO 格式：`类别序号 中心点x 中心点y 宽 高`，后四项为 0~1 归一化值。

数据集未上传，按上述结构可本地还原。

## 5. 训练配置

| 参数 | 值 |
|---|---|
| 模型 | YOLO26s（ultralytics 8.4.213） |
| 预训练权重 | `yolo26s.pt`（COCO 预训练） |
| epochs | 60（跑满，未触发早停） |
| batch-size | 8 |
| imgsz | 640 |
| optimizer | auto |
| seed | 0（固定，可复现） |
| 硬件 | RTX 5060 8GB |
| 环境 | Python 3.13.15 / PyTorch 2.14.1+cu132 |

训练命令：

```bash
D:\yolo-env\Scripts\python.exe scripts\train.py --data datasets/xlsp2/xlsp2.yaml --epochs 60 --batch 8 --imgsz 640 --name xlsp2
```

## 6. 验证集指标

| 类别 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.988 | 1.000 | 0.995 | 0.685 |

第 1 轮 mAP50 仅 0.4975，第 60 轮达到 0.995，训练过程收敛正常。

## 7. 遇到的问题

1. **`head` 类别无法用 COCO 预标注**：COCO 里没有「人头」这个类，直接用检测模型预标注会完全漏掉。改用 pose 关键点模型先找面部关键点、再反推头部框，绕开了这个问题。

2. **双类别容易数量对不上**：如果 person 和 head 分别标注，很容易出现某张图只有 person 没 head。解决方案是标注后统一校验每张图两类数量是否相等，不相等就说明有漏标。

3. **竖屏视频尺寸特殊**：1920×1080 竖屏意味着 imgsz=640 会做较大比例缩放，远处小目标容易被压没。好在本视频人物较大、距离近，影响有限。

4. **数据量小**：只有 46 张图、92 个框，mAP50-95 只有 0.685，比 mAP50 低不少，说明框的精度还有提升空间。样本量是目前最大的瓶颈。

## 8. 复现步骤

```bash
# 1. 环境
python -m venv D:\yolo-env
pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu132
pip install ultralytics pyyaml

# 2. 抽帧
python scripts/extract_frames.py --video raw_video/xlsp2.mp4 --out datasets/xlsp2 --interval 0.25

# 3. person 类自动预标注
python scripts/prelabel.py --root datasets/xlsp2

# 4. 校验
python scripts/check_dataset.py --yaml datasets/xlsp2/xlsp2.yaml

# 5. 训练
python scripts/train.py --data datasets/xlsp2/xlsp2.yaml --epochs 60 --batch 8 --imgsz 640 --name xlsp2

# 6. 推理
python scripts/make_result.py --weights runs/train/xlsp2/weights/best.pt --video raw_video/xlsp2.mp4 --imgsz 640
```

## 9. 交付物

| 文件 | 说明 |
|---|---|
| `result.mp4` | 检测结果视频，1080×1920 / 30fps / 12.4 秒 |
| `frame_01.png` ~ `frame_03.png` | 从结果视频截取的 3 帧 |
| `best.pt` | 训练产出的权重 |
| `README.md` | 本文件 |

## 10. 提交前自检

- [x] train / val 无重叠
- [x] 标注框数值合法
- [x] README 中类别数量（nc=2）与实际标注一致
- [x] person / head 两类数量完全相等（46 : 46）
- [x] result.mp4 时长 12.4 秒，在 10~30 秒内
- [x] 单文件均小于 50 MB（最大 23.5 MB）
