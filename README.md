# ONNX to HuskyLens 2 Installation Package GUI Tool

Convert ONNX models into kmodel format and package them into HuskyLens 2 ZIP installation packages.

* [中文版本](./README_CN.md)

## Features

-  **Supports YOLOv8n, YOLO11n, object detection, semantic segmentation, and object classification models**
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
mean = [0, 0, 0]
std = [1, 1, 1]
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

## Onnx2kmodel Usage Instructions
This tool can convert ONNX models into the kmodel format specifically used by HUSKYLENS 2. And ONNX models can be derived from YOLOv8N and YOLO11N models.
Here are two methods of training two YOLO models, converting them into ONNX models and ultimately generating the HUSKYLENS 2 installation package.
Users can choose either one at will.

### 1. Create the installation package for HUSKYLENS 2 based on Mind+
This method use [Mind+ V2](https://mindplus.cc/en/) to train the yolo mode, and convert the yolo model into onnx model in Mind+ V2. 
For the detail description, please visit  [No-Code Model Training and Deployment-Mind+ Local](https://wiki.dfrobot.com/sen0638/docs/22604#1.2%20No-Code%20Model%20Training%20and%20Deployment-Mind%2B%20Local) in HUSKYLENS 2 wiki.

Users who train the model using this method need to confirm the presence of two compressed files before proceeding to the next step:
1. A.zip compressed file exported from Mind+ containing the ONNX model
2. A.zip compressed file of the dataset

#### Start the "onnx2kmodel" program to convert the ONNX model to a Kmodel model.

The general process is as follows:
* Execute the python app.py file at the project's top level.
* Select the mode - choose MindPlus.
* Select the model package exported by MindPlus.
* Select the dataset package exported by MindPlus.
* Select your own icon.
* Enter the application name in multiple languages (required).
* Enter the title name in multiple languages (required).
* Set a reasonable default output threshold.
* Click "Save Configuration" to set it as the default configuration for opening the GUI tool again (optional).
* Click the "Convert & Package" button. After a few minutes (depending on your computer's performance), a zip format installation package will be generated in the same directory as app.py (note: do not change the name of this installation package).

###  2. Create HUSKYLENS 2 installation package based on custom data

For experienced users familiar with YOLO dataset structure. No further explanation will be given regarding the format of the dataset.
You can visit [Code-Trained Model Deployment to HUSKYLENS 2](https://wiki.dfrobot.com/sen0638/docs/22604#1.3%20Code-Trained%20Model%20Deployment%20to%20HUSKYLENS%202) on HUSKYLENS 2 wiki. Use the sample dataset provided on the wiki and refer to the tutorial link for training the YOLO model and converting it to an ONNX model.

Users who train the model using this method need to confirm the existence of two files before proceeding to the next step:
1. ONNX model
2. Dataset folder

#### Make a folder required for converting the kmodel model to ONNX format

After obtaining the ONNX model, users need to create the following file structure in the same directory as app.py.
Here:
- **best.onnx**: This model is the model generated by converting the YOLO model in the previous step
- **images**: This folder contains the original dataset
- **data.yaml**: The example of this file can be find in folder [examples_yaml](/examples_yaml/). The "names" tag needs to be modified according to the specifications of your model. Other parameters should remain unchanged by default.

Detection and Segmentation Model
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

Classification Model
```shell
.
└── user_dir
    ├── best.onnx
    ├── data.yaml
    └── images
        └── train
            └── cls1    		
                ├── capture_f845db40.png
                ├── capture_fc0e6b54.png
                └── ......
            └── cls2
                ├── capture_fc577b9b.png
                ├── capture_fe2a84a1.png
                └── ......
            └── ...
```

## HuskyLens V2 Installation Package

* Copy the generated ZIP package to:Huskylens\storage\installation_package on the HuskyLens MTP device.
* On HuskyLens, open Model Installation → Local Installation.The app will install automatically.
* Return to the main menu to find and launch your new app.

## Known Issues

* The GUI becomes unresponsive during model conversion (conversion runs on the main thread).
