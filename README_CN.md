# ONNX 转 二哈2安装包 GUI工具

将onnx模型量化为kmodel模型，再将其打包为二哈2的zip安装包

* [English Version](./README.md)

## 功能特性

-  **支持yolov8n目标检测**
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

## 运行GUI程序

```shell
python app.py
```

## GUI工具使用流程

### 基于Mind+制作二哈安装包

* 模式选择 选择MindPlus
* 选择Mindplus导出的模型包
* 选择MindPlus导出的数据集包
* 选择自己的图标
* 输入多语言的应用名称（必填）
* 输入多语言的title名称（必填）
* 设置合理的默认输出阈值
* 点击保存配置，可以作为再次打开gui工具的默认配置（可选）
* 点击转换&打包按钮，等待几分钟（依据你的电脑性能）后，app.py的同级目录会生成一个zip格式的安装包（注意不要更改这个安装包的名字）

###  基于自定义数据制作二哈安装包

假设使用此功能的用户比较了解yolo数据集，这里不对数据集的格式做更多解释

#### 目录结构

用户在app.py的同级目录下，创建如下文件结构

```shell
.
└── user_dir
    ├── best.onnx
    ├── data.yaml
    ├── images
    │   └── train
    │       ├── capture_f845db40.png
    │       ├── capture_fc0e6b54.png
    │       ├── capture_fc577b9b.png
    │       ├── capture_fe2a84a1.png
    │       └── ......
    └── labels
        └── train
            ├── capture_f845db40.txt
            ├── capture_fc0e6b54.txt
            ├── capture_fc577b9b.txt
            ├── capture_fe2a84a1.txt
            └── ......

```

## 二哈2上安装应用

* 将zip安装包拷贝到二哈MTP设备的 Huskylens\storage\installation_package  目录
* 打开二哈 模型安装（Model Installation），选择本地安装（Local Installation），应用就安装好了，回到主界面可以查看

## 遗留问题

* 点击转换时，GUI线程会卡住，转换完成后才可继续操作
* 不支持多语言标签
