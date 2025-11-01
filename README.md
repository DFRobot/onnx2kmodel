# ONNX to HuskyLens 2 Installation Package GUI Tool

Convert ONNX models into kmodel format and package them into HuskyLens 2 ZIP installation packages.

* [中文版本](./README_CN.md)

## Features

-  **Supports YOLOv8n object detection**
-  **Supports Python 3.10 and above** 
-  **Runs on Windows 10, Linux** 
-  **Quantization using training dataset images** 
-  **Uses uint8 quantization** 

## Installation

### Clone the Repository

```
git clone https://github.com/DFRobot/onnx2kmodel
```

### Install Dependencies

#### windows

Download and install .NET 7.0:

https://downloadcd.dfrobot.com.cn/HUSKYLENS/dotnet-sdk-7.0.410-win-x64.exe

Ensure Python 3.10 or higher is installed, then run:

```shell
pip install  -r requirements.txt
pip install  nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl
```

#### Linux

安装 dotnet7.0

Install .NET 7.0 (tested on Ubuntu 22.04):

```shell
sudo apt update
sudo apt install -y dotnet-sdk-7.0
```

Install Python dependencies:

```shell
pip install  -r requirements.txt
pip install  nncase-kpu
```

#### Mac

```
Not yet tested.
```



## Configuration Files

#### Application Configuration -- app_conf.toml

Click “Save Config” in the GUI to automatically update this file.
It will be used as the default configuration next time you open the UI.

```toml
[comm]
mode = "MindPlus"    #use models/datasets exported from MindPlus, mode="User"  for user directory
icon_file = ""       #Required PNG icon with transparent background for packaging
app_name_EN = "Cell\\nRecognition"  # App name (English), "\n" for line break
app_name_zh_CN = "细胞识别"			  #App name (Simplified Chinese)
app_name_zh_TW = "細胞識別"			  #App name (Traditional Chinese)
title_name_EN = "Cell Recognition"  #Title shown in HuskyLens (English)
title_name_zh_CN = "细胞识别"         #Title shown in HuskyLens (Simplified Chinese)
title_name_zh_TW = "細胞識別"         #Title shown in HuskyLens (Traditional Chinese)
det_threshold = 0.6                 #Default detection threshold (0–1)

[mindplus_options]
dataset_zip = ""   # Dataset ZIP exported from MindPlus
model_zip = ""     # Model ZIP exported from MindPlus

[user_options]
user_dir = ""   # Directory for custom files in User mode

```



#### Model Configuration  kmodel_conf.toml

For advanced users familiar with nncase.
Beginners can use the default settings without modification.

```toml
[compile_options]
target = "k230"  # "cpu"
dump_ir = false
dump_asm = false
dump_dir = "./dump"
input_file = ""
preprocess = true
input_type = "uint8"  # "uint8", "float32"
input_shape = [1, 3, 320, 320]
input_range = [0, 1]
input_layout = "NCHW"  # "NHWC"
swapRB = false
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
letterbox_value = 0
output_layout = "NCHW"  # "NHWC"

[ptq_options]
calibrate_method = "NoClip"  # "Kld", "NoClip"
finetune_weights_method = "NoFineTuneWeights"
quant_type = "uint8"  # "float32", "int8", "int16"
w_quant_type = "uint8"  # "float32", "int8", "int16"
dump_quant_error = false
dump_quant_error_symmetric_for_signed = false
quant_scheme = ""
quant_scheme_strict_mode = false
export_quant_scheme = false
export_weight_range_by_channel = false
```

## Running the GUI

```shell
python app.py
```

## GUI Workflow

### Create HuskyLens Installation Package (MindPlus Mode)

* Select MindPlus mode.
* Choose the model ZIP exported from MindPlus.
* Choose the dataset ZIP exported from MindPlus.
* Select an icon (PNG with transparent background).
* Enter multi-language app names (required).
* Enter multi-language title names (required).
* Set a reasonable default detection threshold.
* (Optional) Click Save Config to use as the default next time.
* Click Convert & Package — after a few minutes (depending on your hardware),
a ZIP installation package will appear in the same directory as app.py.
⚠️ Do not rename this zip file.

###  Create HuskyLens Installation Package (User Mode)

For experienced users familiar with YOLO dataset structure.

#### Directory Layout Example

In the same directory as app.py, create the following structure:

```shell
.
└── user_dir
    ├── best.onnx
    ├── data.yaml
    └── images
        └── train
            ├── capture_f845db40.png
            ├── capture_fc0e6b54.png
            ├── capture_fc577b9b.png
            ├── capture_fe2a84a1.png
            └── ......

```

## Installing the Application on HuskyLens 2

* Copy the generated ZIP package to:Huskylens\storage\installation_package on the HuskyLens MTP device.
* On HuskyLens, open Model Installation → Local Installation.The app will install automatically.
* Return to the main menu to find and launch your new app.

## Known Issues

* The GUI becomes unresponsive during model conversion (conversion runs on the main thread).
* Multi-language labels are not yet supported.
