# HUSKYLENS 2 Model Installation Package Generator

Convert supported Ultralytics ONNX models to K230 `.kmodel` files and generate ZIP installation packages that can be installed locally on HUSKYLENS 2.

[简体中文](./README_CN.md)

## Features

- Supports `YOLOv8n` and `YOLO11n`.
- Supports detection, classification, and segmentation tasks.
- Supports `224 × 224`, `320 × 320`, and `640 × 640` input sizes.
- Supports YOLO and MindPlus data sources, with YOLO selected by default.
- Automatically validates and organizes ONNX models, dataset configurations, training images, and labels.
- Selects balanced calibration images from the training set and performs UINT8 quantization with nncase.
- Supports UI and app names in English, Simplified Chinese, Traditional Chinese, French, Korean, Spanish, Brazilian Portuguese, and Japanese.
- Runs conversion and packaging in a separate process while displaying the current stage in real time.
- Lets users choose the installation package output directory and safely manage temporary files.

Note: Do not modify a generated ZIP package in any way. Changing either the files inside the ZIP or the ZIP filename will prevent HUSKYLENS 2 from recognizing the installation package.

## Requirements

### Windows

- Python 3.10 or later. Development and testing use Python 3.12.
- [.NET SDK 7.0.410](https://dotnet.microsoft.com/en-us/download/dotnet/7.0).

Install the Python dependencies:

```powershell
pip install -r requirements.txt
pip install nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl
```

The distributed EXE includes its Python runtime, so end users do not need to install Python separately. The required .NET environment must still be installed.

### Linux

The GUI and nncase dependencies must be validated by the user. On Ubuntu, install .NET 7 and the Python dependencies with:

```bash
sudo apt update
sudo apt install -y dotnet-sdk-7.0
pip install -r requirements.txt
pip install nncase-kpu
```

macOS has not been validated.

## Start

Run from source:

```powershell
python app.py
```

Startup defaults:

- The UI starts in English and can be changed at the top of the window.
- The default data source is YOLO.
- The default threshold is `0.60`, with a step of `0.02`.

## Workflow

1. Select the UI language.
2. Select the installation package output directory.
3. Select the data source (YOLO by default).
4. For YOLO, select the dataset folder and ONNX model separately. For MindPlus, select the model package (.zip) and dataset package (.zip).
5. Review the automatically detected model information and manually select any field that cannot be identified reliably.
6. Enter an app name. Press Enter to add a line break.
7. Add other language names, replace the icon, or adjust the default threshold as needed.
8. When the status area displays “Preparation complete. Ready to start,” click **Start**.
9. Review the installation package path in the completion dialog. After the dialog is closed, the temporary cache from this run is cleared automatically.

## Data Sources

### YOLO: Detection and Segmentation

YOLO mode requires a dataset folder and an `.onnx` model to be selected separately. The root of the dataset folder must contain:

- One `.yaml` or `.yml` dataset configuration file.

The configuration must contain non-empty `train` and `names` fields. The `path` field is optional. When `path` is omitted, the selected dataset folder (the YAML root) is used as the base for resolving `train`. When `path` is present, the training image directory is resolved from `path + train`. The program then locates the label directory by replacing the last `images` component in the training path with `labels`.

Example dataset folder:

```text
dataset_dir
├── data.yaml
├── images
│   ├── train
│   │   ├── image_001.jpg
│   │   └── ...
│   └── val                     # Optional; not used during conversion
│       └── ...
└── labels
    ├── train
    │   ├── image_001.txt
    │   └── ...
    └── val                     # Optional; not used during conversion
        └── ...
```

Example `data.yaml`:

```yaml
path: ./             # Optional
train: ./images/train
val: ./images/val    # Optional; not used during conversion
names:
  0: person
  1: car
```

If multiple `.yaml` files exist at the root of the dataset folder, the program requires all but one correct configuration file to be removed.

### YOLO: Classification

A YAML file is optional for classification. If the ONNX model can be reliably identified as a classification model, the standard directory structure can be used directly:

```text
dataset_dir
├── train
│   ├── class_1
│   │   ├── image_001.jpg
│   │   └── ...
│   └── class_2
│       ├── image_002.jpg
│       └── ...
└── val                       # Optional; not used during conversion
    └── ...
```

The program sorts class directory names to generate `data.yaml`. If the ONNX classification output count can be confirmed, it is checked against the number of classes in the dataset, and preparation stops when the counts differ. Class names and order are not compared between ONNX metadata and the dataset; names come from the dataset YAML or class directories. If the ONNX class count cannot be confirmed, preparation may continue, but the user is prompted to verify that the dataset classes match those used during training.

If the classification folder contains a valid YAML file, its class order and training path take precedence. When `path` is omitted, `train` is resolved directly from the dataset folder.

### MindPlus

Select both:

1. The model package (.zip) exported by MindPlus.
2. The dataset package (.zip) exported by MindPlus.

## ONNX Model Information Detection

The YOLO data source displays model version, task type, and input size controls:

- Model Version: Unknown, YOLOv8n, YOLO11n.
- Task Type: Unknown, Detection, Classification, Segmentation.
- Input Size: Unknown, 224 × 224, 320 × 320, 640 × 640.

The program uses a conservative detection strategy:

1. It first reads ONNX metadata such as `description`, `model_name`, `task`, `head`, `imgsz`, and `input_shape`.
2. If the size is absent from the metadata, it reads the ONNX input tensor.
3. If the task type is absent from the metadata, it combines the ONNX output structure with limited evidence from the dataset structure and labels.
4. Conflicting or insufficient evidence leaves the field as Unknown instead of guessing.
5. Automatically confirmed fields are locked; fields that cannot be confirmed remain available for manual selection.
6. Preparation stops if the model is explicitly identified as an unsupported version such as YOLOv8s or YOLO11s, or if its input size is unsupported.

## Quantization Calibration Image Selection

Calibration images are selected from the training set after the user clicks **Start**:

- Up to 500 classes: cover all classes, select at most 500 images in total, and allocate up to 10 images per class as evenly as possible.
- More than 500 classes: prioritize one image per class; the total may exceed 500 to cover every class.
- Images are selected randomly to maintain substantial variation among samples.
- One multi-class image can cover multiple classes.

If no usable image can be found for a class, the program displays the specific class and path instead of proceeding directly to model conversion.

## App Name, Preview, and Icon

- When the app name is empty, the preview displays the `App Name` placeholder.
- Manual line breaks: press Enter while entering the app name to insert a line-break marker.
- Automatic wrapping: approximately 12 English letters or 6 Chinese characters per line, with at most two lines displayed.
- Users can add app names in any of the eight supported languages. Languages without a separate value use the current default name.
- A square icon with a transparent background and white line art is recommended for visual consistency with native HUSKYLENS 2 icons, although color icons are also supported. The icon size is 60 × 60 pixels; images in other sizes are automatically resized to 60 × 60.

## Output and Cache

The final ZIP is written to the selected output directory using the following format:

```text
AppName-ModelVersion-TaskType-InputSize.Checksum.zip
```

Example:

```text
Cat-YOLO11n-cls-320.a1b2.zip
```

The temporary directory is located at:

```text
Selected output directory/HUSKYLENS 2 Package Generator Temp Files
├── model_input
├── model_output
└── dump
```

- Calibration images selected for quantization can be viewed in `model_input`.
- Changing the output directory removes the temporary files from the previous output directory.
- Preparing a new data source rebuilds `model_input`.
- `model_output` and `dump` are rebuilt before every conversion.
- After the completion dialog is closed, the temporary cache is cleared automatically without prompting; the final ZIP is preserved.

## Install on HUSKYLENS 2

1. Copy the generated ZIP to the following path on the USB drive exposed by HUSKYLENS 2:

   ```text
   Huskylens\storage\installation_package
   ```

2. Open **Model Installation** on HUSKYLENS 2.
3. Select **Local Installation** to complete the installation.

## Configuration Files

### Application Configuration: app_conf.toml

```toml
[comm]
mode = "User"               # Data source: User means YOLO; MindPlus means MindPlus
icon_file = ""              # App icon path; bundled icon.png is used by default at startup
det_threshold = 0.6          # Default threshold, displayed as 0.60 in the UI
app_name_en = ""            # English app name; \\n represents a line break
app_name_zh-CN = ""         # Simplified Chinese app name
app_name_zh-TW = ""         # Traditional Chinese app name
app_name_fr = ""            # French app name
app_name_ko = ""            # Korean app name
app_name_es = ""            # Spanish app name
app_name_pt-BR = ""         # Brazilian Portuguese app name
app_name_ja = ""            # Japanese app name
title_name_en = ""          # English title
title_name_zh-CN = ""       # Simplified Chinese title
title_name_zh-TW = ""       # Traditional Chinese title
title_name_fr = ""          # French title
title_name_ko = ""          # Korean title
title_name_es = ""          # Spanish title
title_name_pt-BR = ""       # Brazilian Portuguese title
title_name_ja = ""          # Japanese title
user_added_langs = []        # Additional app-name languages selected by the user

[mindplus_options]
dataset_zip = ""            # MindPlus dataset package (.zip)
model_zip = ""              # MindPlus model package (.zip)

[user_options]
user_dir = ""               # YOLO dataset folder
onnx_file = ""              # YOLO ONNX model file
```

### Model Configuration: kmodel_conf.toml

`kmodel_conf.toml` contains the nncase compilation and UINT8 quantization settings. Keeping the default values is recommended for most users. During conversion, the program automatically sets the actual input size from the selected ONNX model, so `input_shape` in this file is only a default template value.

```toml
[compile_options]
target = "k230"             # Compilation target
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
