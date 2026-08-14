# ONNX 转 二哈2安装包 GUI工具

将onnx模型量化为kmodel模型，再将其打包为二哈2的zip安装包

* [English Version](./README.md)

## 功能特性

-  **支持yolov8n yolo11n 目标检测 语义分割 物体分类 三种模型(imgsz 320 或 640)**
-  **支持python 3.10 及以上环境** 
-  **可以运行在win10 linux mac多系统上** 
-  **使用训练集的图片量化** 
-  **使用uint8量化** 

## 安装

### 克隆工程

```
git clone https://github.com/DFRobot/onnx2kmodel
```

### 安装依赖

#### windows

下载并安装dotnet7.0

https://downloadcd.dfrobot.com.cn/HUSKYLENS/dotnet-sdk-7.0.410-win-x64.exe

确保电脑安装了Python3.10或以上版本，命令行运行

```shell
pip install  -r requirements.txt
pip install  nncase_kpu-2.10.0-py2.py3-none-win_amd64.whl
```

#### Linux

安装 dotnet7.0

Ubuntu（测试版本22.04）

```shell
sudo apt update
sudo apt install -y dotnet-sdk-7.0
```

安装依赖包

```shell
pip install  -r requirements.txt
pip install  nncase-kpu
```

#### Mac

```
暂未测试
```



## 配置文件

#### 应用配置文件 app_conf.toml

点击ui页面的 保存配置 ，可以自动更新这个toml文件，成为再次打开ui页面的默认配置

```toml
[comm]
mode = "MindPlus"    #User  MindPlus表示加载MindPlus训练出的模型和导出的数据集   User表示用户配置自己的文件结构
icon_file = ""       #打包二哈安装包必备的图标，png格式，背景透明色
app_name_EN = "Cell\\nRecognition"  #安装时显示的英文名称，\n表示换行，将这个名字换成自己的应用名称
app_name_zh_CN = "细胞识别"			  #安装时显示的简体名称，\n表示换行，将这个名字换成自己的应用名称
app_name_zh_TW = "細胞識別"			  #安装时显示的繁体名称，\n表示换行，将这个名字换成自己的应用名称
title_name_EN = "Cell Recognition"  #二哈打开应用时，title显示的英文名称
title_name_zh_CN = "细胞识别"         #二哈打开应用时，title显示的简体中文名称
title_name_zh_TW = "細胞識別"         #二哈打开应用时，title显示的繁体中文名称
det_threshold = 0.6                 #默认检测阈值 范围 0 - 1

[mindplus_options]
dataset_zip = ""   #MindPlus导出的数据集文件，zip格式
model_zip = ""     #MindPlus导出的模型文件，zip格式

[user_options]
user_dir = ""   #用户模式下，用户村子自定义文件的目录

```



#### 模型配置文件 kmodel_conf.toml

参考佳楠nncase相关文档，初级用户可不更改此文件，直接使用默认配置

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

## 运行GUI程序

```shell
python app.py
```

## onnx2kmodel使用说明
该工具可将onnx模型转化为二哈识图2专用的kmodel模型格式。而onnx模型可由yolov8n, yolo11n模型转化而来。<br/>
以下为两个训练yolo模型，转化为onnx模型最终生成二哈识图安装包的过程。<br/>
使用者可任选其一<br/>
当使用方法2自行训练yolo模型时，imgsz参数可选320或者640两种规格。

### 1. 基于Mind+制作二哈安装包
此方法使用[Mind+ V2](https://mindplus.cc/download.html)来训练yolo模型，并且在Mind+中直接转化为onnx模型<br/>
可访问[二哈识图2 Mind+训练模型并离线部署章节](https://wiki.dfrobot.com.cn/_SKU_SEN0638_Gravity_HUSKYLENS_2_AI_Camera_Vision_Sensor#8.2%20Mind%2B%E6%97%A0%E4%BB%A3%E7%A0%81%E6%96%B9%E5%BC%8F%E8%AE%AD%E7%BB%83%E5%B9%B6%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B(%E6%9C%AC%E5%9C%B0))以查看详细说明。

使用该方式训练模型的用户，执行下一步前需要确认有两个压缩包：
1. 从Mind+导出含有onnx模型的.zip压缩包
2. 数据集的.zip压缩包

#### 启动onnx2kmodel，进行onnx模型到kmodel模型转换

简要流程如下：
* 该项目顶层文件执行python app.py
* 模式选择 选择MindPlus
* 选择Mindplus导出的模型包
* 选择MindPlus导出的数据集包
* 选择自己的图标
* 输入多语言的应用名称（必填）
* 输入多语言的title名称（必填）
* 设置合理的默认输出阈值
* 点击保存配置，可以作为再次打开gui工具的默认配置（可选）
* 点击转换&打包按钮，等待几分钟（依据你的电脑性能）后，app.py的同级目录会生成一个zip格式的安装包（注意不要更改这个安装包的名字）

###  2. 基于自定义数据制作二哈安装包

此方法需要用户自行准备yolo格式的数据集<br/>
假设使用此功能的用户比较了解yolo数据集，这里不对数据集的格式做更多解释
可访问[二哈识图2 Python代码训练模型离线部署章节](https://wiki.dfrobot.com.cn/_SKU_SEN0638_Gravity_HUSKYLENS_2_AI_Camera_Vision_Sensor#8.3%20Python%20%E4%BB%A3%E7%A0%81%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E5%B9%B6%E9%83%A8%E7%BD%B2(%E6%9C%AC%E5%9C%B0))。<br/>
使用我们提供的示例数据集，并参考该链接教程进行yolo模型训练以及onnx模型转化。

使用该方式训练模型的用户，执行下一步前需要确认有两个文件：
1. onnx模型
2. yolo数据集文件夹


##### 制作onnx转换kmodel模型所需文件夹

获得onnx模型后，用户需在app.py的同级目录下，创建如下文件结构。
其中：
- **best.onnx**：该文件是上一步由yolo模型转化生成的onnx模型
- **images**：该文件夹包含了原始数据集
- **data.yaml**：该文件可在[examples_yaml](/examples_yaml/)文件夹中找到示例。其中的names标签，需要根据自己模型的规格进行修改。其他参数默认不动

检测和分割模型
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

分类模型

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

#### 启动onnx2kmodel，进行onnx模型到kmodel模型转换

简要流程如下：
* 该项目顶层文件夹，开启终端执行python app.py
* 模式选择 选择自定义
* 用户自定义目录，选择前一步整理好的user_dir文件夹
* 选择自己的图标
* 输入多语言的应用名称（必填）
* 输入多语言的title名称（必填）
* 设置合理的默认输出阈值
* 点击保存配置，可以作为再次打开gui工具的默认配置（可选）
* 点击转换&打包按钮，等待几分钟（依据你的电脑性能）后，app.py的同级目录会生成一个zip格式的安装包（注意不要更改这个安装包的名字）

## 二哈2上安装应用

* 将zip安装包拷贝到二哈MTP设备的 Huskylens\storage\installation_package  目录
* 打开二哈 模型安装（Model Installation），选择本地安装（Local Installation），应用就安装好了，回到主界面可以查看

## 遗留问题

* 点击转换时，GUI线程会卡住，转换完成后才可继续操作
