# HUSKYLENS 2 模型安装包生成器

将受支持的 Ultralytics ONNX 模型转换为 K230 `.kmodel`，并生成可在 HUSKYLENS 2 中本地安装的 ZIP 安装包。

[English](./README.md) 

关于工具的详细使用教程，请参考：[HUSKYLENS 2 模型安装包生成器使用教程](https://wiki.dfrobot.com/sen0638/docs/22604)。

## 功能概览

- 支持 `YOLOv8n`、`YOLO11n`。
- 支持检测、分类、分割任务。
- 支持 `224 × 224`、`320 × 320`、`640 × 640` 输入尺寸。
- 数据来源支持 YOLO 和 MindPlus，默认选择 YOLO。
- 自动检查并整理 ONNX、数据集配置、训练图片和标签。
- 从训练集均衡抽取校准图片，使用 nncase 执行 UINT8 量化。
- 支持 English、简体中文、繁體中文、Français、한국어、Español、Português (Brasil)、日本語八种界面及应用名称。
- 转换、打包在独立进程中执行，界面实时显示当前阶段。
- 支持选择安装包输出目录并安全管理临时文件。

注意：不可对转换好的ZIP文件做任何修改，包括更改ZIP内部文件内容，或修改文件名，否则都将导致HUSKYLENS 2无法识别安装包。

## 环境要求

### Windows

- Python 3.10 或更高版本；本项目开发测试使用 Python 3.12。
- [.NET SDK 7.0.410](https://dotnet.microsoft.com/en-us/download/dotnet/7.0)。

安装 Python 依赖：

```powershell
pip install -r requirements.txt
pip install nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl
```

如果使用仓库提供的 EXE，用户不需要另外安装 Python，但仍需安装程序依赖的 .NET 环境。

### Linux

图形界面和 nncase 依赖需由使用者自行验证。Ubuntu 可先安装 .NET 7：

```bash
sudo apt update
sudo apt install -y dotnet-sdk-7.0
pip install -r requirements.txt
pip install nncase-kpu
```

macOS 尚未完成验证。

## 启动

源码运行：

```powershell
python app.py
```

默认状态：

- 界面语言为 English，可在顶部切换。
- 数据来源默认为 YOLO。
- 默认阈值为 `0.60`，步进为 `0.02`。

## 使用流程

1. 选择界面语言。
2. 选择安装包输出目录。
3. 选择数据来源（默认 YOLO）。
4. YOLO 分别选择数据集文件夹和 ONNX 模型；MindPlus 分别选择模型包（.zip）和数据集包（.zip）。
5. 检查自动识别出的模型信息；无法可靠识别的项目需手动选择。
6. 输入应用名称，可按 Enter 键添加换行。
7. 按需添加其他语言、更换图标和调整默认阈值。
8. 状态栏显示“准备工作已完成，可以开始”后，点击“开始”。
10. 在完成弹窗中查看安装包路径；关闭弹窗后程序自动清除本次任务缓存。

## 数据来源

### YOLO：检测与分割

YOLO 模式需要分别选择一个数据集文件夹和一个 `.onnx` 模型文件。数据集文件夹根层必须包含：

- 一个 `.yaml` 或 `.yml` 数据集配置文件。

配置文件必须包含非空的 `train` 和 `names`；`path` 可以省略。省略 `path` 时，程序以用户选择的数据集文件夹（YAML 所在根目录）为基准，通过 `train` 定位训练图片；存在 `path` 时使用 `path + train`。随后将训练路径中最后一个 `images` 替换为 `labels` 定位标签。

数据集文件夹示例：

```text
dataset_dir
├── data.yaml
├── images
│   ├── train
│   │   ├── image_001.jpg
│   │   └── ...
│   └── val                     # 可选，转换不读取
│       └── ...
└── labels
    ├── train
    │   ├── image_001.txt
    │   └── ...
    └── val                     # 可选，转换不读取
        └── ...
```

data.yaml示例：

```yaml
path: ./             # 可选
train: ./images/train
val: ./images/val    # 可选，转换不读取
names:
  0: person
  1: car
```

数据集文件夹根层存在多个 .yaml文件时，会要求只保留一个正确文件。

### YOLO：分类

分类任务允许不提供 YAML。ONNX 能明确识别为分类模型时，可直接使用标准目录：

```text
dataset_dir
├── train
│   ├── class_1
│   │   ├── image_001.jpg
│   │   └── ...
│   └── class_2
│       ├── image_002.jpg
│       └── ...
└── val                       # 可选，转换不读取
    └── ...
```

程序按类别目录名称排序生成 `data.yaml`。如果 ONNX 能确认分类输出数量，程序会与数据集类别数量核对，数量不一致时停止准备。程序不比较 ONNX 元数据与数据集中的类别名称和顺序，类别名称以数据集 YAML 或类别目录为准。无法从 ONNX 确认类别数量时允许继续，但会提示用户确认数据集类别与训练时一致。

如果分类目录本身提供了一个有效 YAML，程序仍优先使用该配置的类别顺序和训练路径；未提供 `path` 时直接基于 `train` 定位。

### MindPlus 

需要选择：

1. MindPlus 导出的模型包（.zip）。
2. MindPlus 导出的数据集包（.zip）。

## ONNX 模型信息识别

YOLO 数据源界面显示模型版本、任务类型和输入尺寸：

- 模型版本：未知、YOLOv8n、YOLO11n。
- 任务类型：未知、检测、分类、分割。
- 输入尺寸：未知、224 × 224、320 × 320、640 × 640。

程序采用保守识别策略：

1. 优先读取 ONNX 的 `description`、`model_name`、`task`、`head`、`imgsz`、`input_shape` 等元数据。
2. 元数据没有尺寸时，读取 ONNX 输入张量。
3. 元数据没有任务类型时，结合 ONNX 输出结构和少量数据集结构/标签证据。
4. 多项证据冲突或不足时保持“未知”，不强行猜测。
5. 自动确认的项目会锁定；无法确认的项目允许用户手动选择。
6. 明确识别为 YOLOv8s、YOLO11s 等不支持型号，或输入尺寸不受支持时，停止准备。

## 量化校准图片抽取

点击“开始”后从训练集抽取校准图片：

- 类别不超过 500：覆盖全部类别，总量最多 500 张，每类最多 10 张并均衡分配。
- 类别超过 500：每类优先 1 张，为覆盖全部类别可超过 500 张。
- 图片随机抽取，确保抽取的图片之间会有较大差异。
- 一张多类别图片可同时覆盖多个类别。

如果某个类别没有可用图片，程序会显示具体类别和路径，不会直接进入模型转换。

## 应用名称、预览和图标

- 应用名称为空时，默认显示 `App Name`占位。
- 手动换行：输入框输入应用名称时，按 Enter 会插入换行标记。
- 自动换行：英文每行最多约 12 个字母，中文每行最多约 6 个文字，最多显示两行。
- 用户可添加八种语言中的其他语言名称；未单独填写的语言使用当前默认名称补齐。
- 应用图标建议使用正方形、透明背景、白色线条的图标，这样会与HUSKYLESN 2的原生图标具有较强的一致性，也支持使用彩色图标。图标尺寸为60x60像素，若选择了其他尺寸的图片，会自动处理为60x60。

## 输出与缓存

最终 ZIP 输出到用户选择的目录，命名格式为：

```text
应用名-模型版本-任务类型-输入尺寸.校验码.zip
```

例如：

```text
Cat-YOLO11n-cls-320.a1b2.zip
```

临时目录位于：

```text
用户输出目录/HUSKYLENS 2 Package Generator Temp Files
├── model_input
├── model_output
└── dump
```

- 在model_input中可以查看用于量化校准所抽取的图片。
- 切换输出目录时，会删除旧输出目录中的临时文件。
- 每次重新准备数据会重建 `model_input`。
- 每次转换前会重建 `model_output` 和 `dump`。
- 完成弹窗关闭后自动安全清除临时缓存，不再询问用户；最终 ZIP 不会被删除。

## 在 HUSKYLENS 2 中安装

1. 将生成的 ZIP 复制到 HUSKYLENS 2 弹出的U盘如下路径中：

   ```text
   Huskylens\storage\installation_package
   ```

2. 在HUSKYLENS 2 上打开“模型安装 / Model Installation”。
3. 选择“本地安装 / Local Installation”，即可安装完成。

## 配置文件

### 应用配置文件 app_conf.toml

```toml
[comm]
mode = "User"               # 数据来源：User 表示 YOLO，MindPlus 表示 MindPlus
icon_file = ""              # 应用图标路径；启动时默认使用项目内置 icon.png
det_threshold = 0.6          # 默认阈值，界面显示为 0.60
app_name_en = ""            # English 应用名称，\\n 表示换行
app_name_zh-CN = ""         # 简体中文应用名称
app_name_zh-TW = ""         # 繁體中文应用名称
app_name_fr = ""            # Français 应用名称
app_name_ko = ""            # 한국어 应用名称
app_name_es = ""            # Español 应用名称
app_name_pt-BR = ""         # Português (Brasil) 应用名称
app_name_ja = ""            # 日本語应用名称
title_name_en = ""          # English 标题名称
title_name_zh-CN = ""       # 简体中文标题名称
title_name_zh-TW = ""       # 繁體中文标题名称
title_name_fr = ""          # Français 标题名称
title_name_ko = ""          # 한국어标题名称
title_name_es = ""          # Español 标题名称
title_name_pt-BR = ""       # Português (Brasil) 标题名称
title_name_ja = ""          # 日本語标题名称
user_added_langs = []        # 用户添加的其他应用名称语言

[mindplus_options]
dataset_zip = ""            # MindPlus 数据集包（.zip）
model_zip = ""              # MindPlus 模型包（.zip）

[user_options]
user_dir = ""               # YOLO 数据集文件夹
onnx_file = ""              # YOLO ONNX 模型文件
```

### 模型配置文件 kmodel_conf.toml

`kmodel_conf.toml` 保存 nncase 编译和 UINT8 量化参数。普通用户建议保持默认值。转换时程序会根据所选 ONNX 模型自动设置实际输入尺寸，因此文件中的 `input_shape` 只是默认模板值。

```toml
[compile_options]
target = "k230"             # 编译目标
dump_ir = false
dump_asm = false
dump_dir = "./dump"
input_file = ""
preprocess = true
input_type = "uint8"
input_shape = [1, 3, 640, 640]
input_range = [0, 1]
input_layout = "NCHW"
swapRB = false
mean = [0, 0, 0]
std = [1, 1, 1]
letterbox_value = 0
output_layout = "NCHW"

[ptq_options]
calibrate_method = "NoClip"
finetune_weights_method = "NoFineTuneWeights"
quant_type = "uint8"
w_quant_type = "uint8"
dump_quant_error = false
dump_quant_error_symmetric_for_signed = false
quant_scheme = ""
quant_scheme_strict_mode = false
export_quant_scheme = false
export_weight_range_by_channel = false
```

## 
