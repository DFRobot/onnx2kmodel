# -*- coding: utf-8 -*-

import os
import sys
import zipfile
import hashlib
import shutil
import yaml
import json
import re
import copy
import onnx
import tomlkit
import locale
from PIL import Image
from PyQt5.QtWidgets import (
    QFrame, QApplication, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit,QSpacerItem,QSizePolicy,
    QFileDialog, QComboBox, QSlider, QHBoxLayout, QVBoxLayout, QGridLayout,
    QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

lang_id = 0
lang = {
    "name": ["name","名字"],
    "data": ["Model ","数据包"],
    "icon": ["icon","图标"],
    "custom": ["Custom", "自定义"],
    "select_mode": ["Select Mode","选择模式"],
    "custom_directory": ["Custom Directory","用户自定义目录"],
    "select_model_package": ["Select Model Package","选择模型包"],
    "select_dataset_package": ["Select Dataset Package","选择数据集包"],
    "select_icon": ["Select Icon","选择图标"],
    "app_name": ["App Name","应用名称"],
    "simplified_chinese": ["Simplified Chinese","简体中文"],
    "traditional_chinese": ["Traditional Chinese","繁体中文"],
    "title_settings": ["Title Settings","标题设置"],
    "detection_threshold": ["Detection Threshold","识别阈值"],
    "save_config": ["Save Config","保存配置"],
    "convert_and_package": ["Convert and Package","转换&打包"],
    "pack_only": ["Pack Only","仅打包"],
    "app_title": ["Mindplus Model to K230 Installer","Mindplus模型转二哈安装包"],
    "select_custom_directory": ["Select Custom Directory","选择用户自定义目录"],
    "select_zip_file": ["Select ZIP File","选择ZIP文件"],
    "app_name_cannot_be_empty": ["App Name cannot be empty","应用名称不能为空"],
    "title_name_cannot_be_empty": ["Title Name cannot be empty","标题名称不能为空"],
    "converting_please_wait": ["Converting, please wait...","转换中......, 需要几分钟，请耐心等待"],
    "dialog_warning_title": ["Warning", "警告"],
    "conversion_complete_title": ["Conversion Complete", "转换完成"],
    "conversion_complete_message": [
        "Conversion complete!\n{path}",
        "转换完成！\n{path}",
    ],
}

conf_template = {
    "conf": {
        "application": "",
        "defconfig": {"conf_thres": 0.3},
        "infer_isp": {"format": "BG3P", "channel": 3, "width": 864, "height": 486},
        "fps_limit": 15,
        "model_info": [{"name": "object-detection-detector", "filename": ""}],
        "model_attach": {"classes": {"en": [], "zh-CN": [], "zh-TW": []}}
    }
}

desc_template = {
    "desc": {
        "application_name": ["name", "名字", "名字"],
        "application_title": ["title", "抬头", "抬頭"],
        "base_model": "yolov8n-det",
        "stream": True,
        "version": "0.1"
    }
}

mindplus_base_model_to_kmodel_base_model = {
    "yolov8n-cls": ["yolov8n-cls", "object-classification-classifier"],
    "yolov8n-seg": ["yolov8n-seg", "object-segmentation-segment"],
    "yolov8n": ["yolov8n-det", "object-detection-detector"],
    "yolo11n-cls": ["yolo11n-cls", "object-classification-classifier"],
    "yolo11n-seg": ["yolo11n-seg", "object-segmentation-segment"],
    "yolo11n": ["yolo11n-det", "object-detection-detector"],
}

def clean_name(name):
    name = name.replace("\\n", "\n")
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

def extract_zip(zip_path, output_dir="model_input"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    return output_dir

def extract_zip_without_top(zip_path, output_dir="model_input"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            # 跳过空的顶层目录
            parts = member.filename.split('/')
            if len(parts) > 1:
                # 去掉顶层目录
                target_path = os.path.join(output_dir, *parts[1:])
            else:
                target_path = os.path.join(output_dir, parts[0])

            if member.is_dir():
                os.makedirs(target_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())

    print("解压完成！")

def zip_with_md5(source_dir="model_output/", zip_dir="./", base_name="app"):
    temp_zip_path = os.path.join(zip_dir, f"{base_name}.zip")
    with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, source_dir)
                zipf.write(abs_path, arcname=rel_path)
    md5_hash = hashlib.md5()
    with open(temp_zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    md5_str = md5_hash.hexdigest()[:4]
    final_zip_path = os.path.join(zip_dir, f"{base_name}.{md5_str}.zip")
    shutil.move(temp_zip_path, final_zip_path)
    print(f"打包完成: {final_zip_path}")
    return final_zip_path

def get_input_shape(onnx_path):
    # 加载 ONNX 模型
    model = onnx.load(onnx_path)
    # 获取输入节点信息
    input_tensors = model.graph.input
    input_shapes = {}

    for tensor in input_tensors:
        shape = []
        tensor_type = tensor.type.tensor_type
        if tensor_type.HasField('shape'):
            for dim in tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                else:
                    shape.append(None)  # 动态维度
        input_shapes[tensor.name] = shape

    return input_shapes


def get_tensor_shape(tensor):
    shape = []
    tensor_type = tensor.type.tensor_type
    if tensor_type.HasField('shape'):
        for dim in tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            else:
                shape.append(-1)
    return shape


def get_image_size_from_input_shape(shape):
    if len(shape) < 4:
        return []
    if shape[1] in (1, 3) and shape[-1] not in (1, 3):
        return [shape[2], shape[3]]
    if shape[-1] in (1, 3) and shape[1] not in (1, 3):
        return [shape[1], shape[2]]
    return [shape[-2], shape[-1]]


def analyze_onnx_model(onnx_path):
    model = onnx.load(onnx_path)
    input_shape = []
    if model.graph.input:
        input_shape = get_image_size_from_input_shape(get_tensor_shape(model.graph.input[0]))

    outputs = model.graph.output
    output_shapes = [get_tensor_shape(out) for out in outputs]
    base_model = None

    if len(outputs) == 1:
        shape = output_shapes[0]
        if len(shape) == 2:
            base_model = "yolov8n-cls"
        elif len(shape) == 3:
            last_dim = shape[-1]
            if last_dim == 6 or shape[1] > 6 or last_dim > 6:
                base_model = "yolov8n"
    elif len(outputs) == 2:
        shape0 = output_shapes[0]
        shape1 = output_shapes[1]
        if len(shape0) == 3 and len(shape1) == 4:
            base_model = "yolov8n-seg"

    if not base_model:
        raise ValueError(f"无法根据 ONNX 输出结构识别模型类型: {output_shapes}")
    if not input_shape:
        raise ValueError("无法从 ONNX 输入中识别 input_shape")

    return {
        "base_model": base_model,
        "input_shape": input_shape,
    }


def get_name_list_from_data_yaml(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        source_config = yaml.safe_load(f) or {}

    names = source_config.get("names", {})
    if isinstance(names, list):
        name_list = [str(name) for name in names]
    elif isinstance(names, dict):
        def sort_key(key):
            try:
                return int(key)
            except (TypeError, ValueError):
                return key

        name_list = [str(value) for key, value in sorted(names.items(), key=lambda item: sort_key(item[0]))]
    else:
        raise ValueError(f"data.yaml 中的 names 字段格式不支持: {type(names)}")

    return source_config, name_list


def build_model_config_from_custom_dir(model_dataset_dir):
    onnx_path = os.path.join(model_dataset_dir, "best.onnx")
    data_yaml_path = os.path.join(model_dataset_dir, "data.yaml")
    model_info = analyze_onnx_model(onnx_path)
    _, name_list = get_name_list_from_data_yaml(data_yaml_path)

    if model_info["base_model"].endswith("-cls"):
        aitools_id = "ai-tools-classification"
    elif model_info["base_model"].endswith("-seg"):
        aitools_id = "ai-tools-segmentation"
    else:
        aitools_id = "ai-tools-detection"

    return {
        "aitools_id": aitools_id,
        "aitools_version": "0.0.1",
        "description": os.path.basename(os.path.normpath(model_dataset_dir)),
        "base_model": model_info["base_model"],
        "input_shape": model_info["input_shape"],
        "labels": {index: name for index, name in enumerate(name_list)},
    }


def write_model_yaml(model_yaml_path, model_config):
    lines = [
        f"aitools_id: {model_config['aitools_id']}",
        f"aitools_version: {model_config['aitools_version']}",
        f"description: {model_config['description']}",
        f"base_model: {model_config['base_model']}",
        f"input_shape: {json.dumps(model_config['input_shape'], ensure_ascii=False)}",
        "labels:",
    ]
    for index, name in model_config["labels"].items():
        lines.append(f"  {index}: {json.dumps(str(name), ensure_ascii=False)}")

    with open(model_yaml_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


class ConvertThread(QThread):
    finished = pyqtSignal(str)  # 定义信号，执行完毕触发，传递文件路径

    def __init__(self, onnx_path, kmodel_path, dataset_path, conf_path, output_zip_file):
        super().__init__()
        self.onnx_path = onnx_path
        self.kmodel_path = kmodel_path
        self.dataset_path = dataset_path
        self.conf_path = conf_path
        self.output_zip_file = output_zip_file

    def run(self):
        import convertor
        # 耗时操作放在这里
        shapes = get_input_shape(self.onnx_path)
        if 1 == len(shapes):
            shape = list(shapes.values())[0]
        convertor.make(self.onnx_path, self.kmodel_path, self.dataset_path, self.conf_path,shape)
        file_path = zip_with_md5(base_name=self.output_zip_file)
        self.finished.emit(file_path)  # 发射信号通知主线程


class ModelExportApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(lang["app_title"][lang_id])
        self.resize(800, 600)
        with open("app_conf.toml", 'r', encoding='utf-8') as f:
            self._conf = tomlkit.parse(f.read())
            print(self._conf)
        self.separator = {}
        self.init_ui()


    def init_ui(self):
        self.main_layout = QVBoxLayout()

        # --- 第一行：模式选择 ---
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Language:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "中文"])
        self.lang_combo.setCurrentIndex(lang_id)
        self.lang_combo.currentIndexChanged.connect(self.lang_changed)
        lang_layout.addWidget(self.lang_combo)
        self.main_layout.addLayout(lang_layout)


        mode_layout = QHBoxLayout()
        self.select_mode_label = QLabel(lang["select_mode"][lang_id])
        mode_layout.addWidget(self.select_mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["MindPlus", lang["custom"][lang_id]])
        if self._conf["comm"]["mode"] == "MindPlus":
            self.mode_combo.setCurrentIndex(0)
        else:
            self.mode_combo.setCurrentIndex(1)
        self.mode_combo.currentIndexChanged.connect(self.mode_changed)
        mode_layout.addWidget(self.mode_combo)
        self.main_layout.addLayout(mode_layout)


        self.add_separator("select_mode")
        # --- 第二行：模型包选择 ---
        model_layout = QHBoxLayout()
        self.zip_model_button = QPushButton(lang["select_model_package"][lang_id] + " (*.zip)")
        self.zip_model_button.clicked.connect(lambda: self.select_zip("model"))
        self.zip_model_label = QLineEdit(self._conf["mindplus_options"]["model_zip"])
        self.zip_model_label.setReadOnly(True)
        model_layout.addWidget(self.zip_model_button)
        model_layout.addWidget(self.zip_model_label)
        self.main_layout.addLayout(model_layout)

        # --- 第三行：数据包选择 ---
        dataset_layout = QHBoxLayout()
        self.zip_dataset_button = QPushButton(lang["select_dataset_package"][lang_id] + " (*.zip)")
        self.zip_dataset_button.clicked.connect(lambda: self.select_zip("dataset"))
        self.zip_dataset_label = QLineEdit(self._conf["mindplus_options"]["dataset_zip"])
        self.zip_dataset_label.setReadOnly(True)
        dataset_layout.addWidget(self.zip_dataset_button)
        dataset_layout.addWidget(self.zip_dataset_label)
        self.main_layout.addLayout(dataset_layout)

        # --- 自定义目录选择 ---
        user_layout = QHBoxLayout()
        self.user_dir_button = QPushButton(lang["custom_directory"][lang_id])
        self.user_dir_button.clicked.connect(self.select_user_dir)
        self.user_dir_label = QLineEdit(self._conf["user_options"]["user_dir"])
        self.user_dir_label.setReadOnly(True)
        user_layout.addWidget(self.user_dir_button)
        user_layout.addWidget(self.user_dir_label)
        self.main_layout.addLayout(user_layout)

        # --- 图标选择 ---
        self.add_separator("icon")
        icon_layout = QHBoxLayout()
        self.icon_button = QPushButton(lang["select_icon"][lang_id])
        self.icon_button.clicked.connect(self.select_icon)
        self.icon_preview = QLabel()
        if self._conf["comm"]["icon_file"] and os.path.exists(self._conf["comm"]["icon_file"]):
            img = Image.open(self._conf["comm"]["icon_file"])
            if img.size != (60, 60):
                img = img.resize((60, 60))
            if os.path.exists("model_output"):
                img.save("model_output/icon.png")
            pixmap = QPixmap(self._conf["comm"]["icon_file"])
            self.icon_preview.setPixmap(pixmap)
        icon_layout.addWidget(self.icon_button)
        icon_layout.addWidget(self.icon_preview)
        self.main_layout.addLayout(icon_layout)

        # --- 应用名称 ---
        self.add_separator("app_name")
        self.app_zh = QLineEdit(self._conf["comm"]["app_name_zh_CN"])
        self.app_tw = QLineEdit(self._conf["comm"]["app_name_zh_TW"])
        self.app_en = QLineEdit(self._conf["comm"]["app_name_EN"])
        self.app_name_label_zh = QLabel(lang["simplified_chinese"][lang_id])
        self.main_layout.addWidget(self.app_name_label_zh)
        self.main_layout.addWidget(self.app_zh)
        self.app_name_label_tw = QLabel(lang["traditional_chinese"][lang_id])
        self.main_layout.addWidget(self.app_name_label_tw)
        self.main_layout.addWidget(self.app_tw)
        self.app_name_label_en = QLabel("English")
        self.main_layout.addWidget(self.app_name_label_en)
        self.main_layout.addWidget(self.app_en)

        self.add_separator("title_settings")
        self.title_zh = QLineEdit(self._conf["comm"]["title_name_zh_CN"])
        self.title_tw = QLineEdit(self._conf["comm"]["title_name_zh_TW"])
        self.title_en = QLineEdit(self._conf["comm"]["title_name_EN"])
        self.title_name_label_zh = QLabel(lang["simplified_chinese"][lang_id])
        self.main_layout.addWidget(self.title_name_label_zh)
        self.main_layout.addWidget(self.title_zh)        
        self.title_name_label_tw = QLabel(lang["traditional_chinese"][lang_id])
        self.main_layout.addWidget(self.title_name_label_tw)
        self.main_layout.addWidget(self.title_tw)
        self.main_layout.addWidget(QLabel("English"))
        self.main_layout.addWidget(self.title_en)

        # --- 阈值设置 ---
        self.add_separator("detection_threshold")
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(self._conf["comm"]["det_threshold"]*100))
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        self.threshold_label = QLabel(str(self._conf["comm"]["det_threshold"]))
        #threshold_layout.addWidget(QLabel("默认识别阈值"))
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        self.main_layout.addLayout(threshold_layout)

        # --- 底部按钮 ---
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton(lang["save_config"][lang_id])
        self.save_btn.clicked.connect(self.save_conf)
        self.export_btn = QPushButton(lang["convert_and_package"][lang_id])
        self.export_btn.clicked.connect(self.export_model)
        self.pack_btn = QPushButton(lang["pack_only"][lang_id])
        self.pack_btn.clicked.connect(self.pack)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.pack_btn)
        btn_layout.addStretch(1)
        self.main_layout.addLayout(btn_layout)

        # 设置主布局
        self.setLayout(self.main_layout)

        # 初始化控件显隐
        self.mode_changed(self.mode_combo.currentIndex())

    def add_separator(self, key=""):
        title = lang[key][lang_id]
        spacer = QSpacerItem(0, 30, QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.main_layout.addItem(spacer)

        line_layout = QHBoxLayout()    

        line_left = QFrame()
        line_left.setFrameShape(QFrame.HLine)
        line_left.setFrameShadow(QFrame.Sunken)    
        self.separator[key] = QLabel(title)
        self.separator[key].setAlignment(Qt.AlignCenter)    

        font = self.separator[key].font()
        font.setBold(True)
        self.separator[key].setFont(font)

        line_right = QFrame()
        line_right.setFrameShape(QFrame.HLine)
        line_right.setFrameShadow(QFrame.Sunken)    

        line_layout.addWidget(line_left)
        line_layout.addWidget(self.separator[key])
        line_layout.addWidget(line_right)    

        self.main_layout.addLayout(line_layout)

    def update_threshold_label(self, value):
        # 转换为 0.00 ~ 1.00
        f_value = value / 100.0
        self._conf["comm"]["det_threshold"] = float("{:.2f}".format(f_value))
        self.threshold_label.setText(f"{f_value:.2f}")

    def select_zip(self,file_type):
        file, _ = QFileDialog.getOpenFileName(self, lang["select_zip_file"][lang_id], "", "ZIP files (*.zip)")
        if file:
            if file_type == "model":
                self._conf["mindplus_options"]["model_zip"] = file
                self.zip_model_label.setText(file)
            elif file_type == "dataset":
                self._conf["mindplus_options"]["dataset_zip"] = file
                self.zip_dataset_label.setText(file)

    def select_user_dir(self):
        directory = QFileDialog.getExistingDirectory(self, lang["select_custom_directory"][lang_id], "")
        if directory:
            self._conf["user_options"]["user_dir"] = directory
            self.user_dir_label.setText(directory)

    def select_icon(self):
        file, _ = QFileDialog.getOpenFileName(self, lang["select_icon"][lang_id], "", "PNG files (*.png)")
        if file:
            img = Image.open(file)
            img = img.convert('RGBA')
            if img.size != (60, 60):
                base, _ = os.path.splitext(file)
                new_file = f"{base}_60_60.png"
                img = img.resize((60, 60))
                img.save(new_file)
                file = new_file

            self._conf["comm"]["icon_file"] = file
            if os.path.exists("model_output"):
                img.save("model_output/icon.png")
            pixmap = QPixmap(file)
            self.icon_preview.setPixmap(pixmap)

    def save_conf(self):
        self._conf["comm"]["app_name_zh_CN"] = self.app_zh.text()
        self._conf["comm"]["app_name_EN"] = self.app_en.text()
        self._conf["comm"]["app_name_zh_TW"] = self.app_tw.text()

        self._conf["comm"]["title_name_zh_CN"] = self.title_zh.text()
        self._conf["comm"]["title_name_EN"] = self.title_en.text()
        self._conf["comm"]["title_name_zh_TW"] = self.title_tw.text()

        with open("app_conf.toml", 'w', encoding='utf-8') as f:
            f.write(tomlkit.dumps(self._conf))

    def lang_changed(self, index):
        global lang_id
        selected_lang = self.lang_combo.itemText(index)
        print("当前语言:", selected_lang)
        # 你可以根据模式做其他操作
        lang_id = self.lang_combo.currentIndex()
        # 更新按钮文本
        self.setWindowTitle(lang["app_title"][lang_id])
        self.save_btn.setText(lang["save_config"][lang_id])
        self.export_btn.setText(lang["convert_and_package"][lang_id])
        self.pack_btn.setText(lang["pack_only"][lang_id])
        self.select_mode_label.setText(lang["select_mode"][lang_id])
        self.user_dir_button.setText(lang["custom_directory"][lang_id])
        self.icon_button.setText(lang["select_icon"][lang_id])
        self.app_name_label_zh.setText(lang["simplified_chinese"][lang_id])
        self.app_name_label_tw.setText(lang["traditional_chinese"][lang_id])
        self.title_name_label_zh.setText(lang["simplified_chinese"][lang_id])
        self.title_name_label_tw.setText(lang["traditional_chinese"][lang_id])
        self.mode_combo.clear()
        self.mode_combo.addItems(["MindPlus", lang["custom"][lang_id]])
        self.zip_model_button.setText(lang["select_model_package"][lang_id])
        self.zip_dataset_button.setText(lang["select_dataset_package"][lang_id])
        
        # 更新分隔符文本
        for key in self.separator.keys():
            self.separator[key].setText(lang[key][lang_id])

    def mode_changed(self, index):
        selected_mode = self.mode_combo.itemText(index)
        print("当前模式:", selected_mode)
        # 你可以根据模式做其他操作
        if selected_mode == "MindPlus":
            self._conf["comm"]["mode"]  = "MindPlus"
            self.zip_model_button.show()
            self.zip_model_label.show()
            self.zip_dataset_button.show()
            self.zip_dataset_label.show()
            self.user_dir_button.hide()
            self.user_dir_label.hide()
        else:
            print("self._conf[comm][mode]  = 'User'")
            self._conf["comm"]["mode"]  = "User"
            print(self._conf)
            self.zip_model_button.hide()
            self.zip_model_label.hide()
            self.zip_dataset_button.hide()
            self.zip_dataset_label.hide()
            self.user_dir_button.show()
            self.user_dir_label.show()
        self.setLayout(self.main_layout)


    def export_model(self):
        print(self._conf)
        if self._conf["comm"]["mode"] == "MindPlus":
            #制作MindPlus数据目录
            if os.path.exists("model_input"):
                shutil.rmtree("model_input")
            os.makedirs("model_input", exist_ok=True)
            model_zip = self._conf["mindplus_options"]["model_zip"] 
            if not model_zip or not os.path.exists(model_zip):
                print(f"模型包不存在: {model_zip}")
                return
            extract_zip(model_zip, "model_input")
            #将data.yaml 改名为model.yaml
            os.rename(os.path.join("model_input", "data.yaml"), os.path.join("model_input", "model.yaml"))

            dataset_zip = self._conf["mindplus_options"]["dataset_zip"] 
            if not dataset_zip or not os.path.exists(dataset_zip):
                print(f"数据集包不存在: {dataset_zip}")
                return
            #extract_zip_without_top(dataset_zip, "model_input")
            extract_zip(dataset_zip, "model_input")
            self.model_dataset_dir = "model_input"
        else:
            #使用用户自定义目录
            self.model_dataset_dir = self._conf["user_options"]["user_dir"]
            model_yaml_path = os.path.join(self.model_dataset_dir, "model.yaml")
            if not os.path.exists(model_yaml_path):
                model_config = build_model_config_from_custom_dir(self.model_dataset_dir)
                write_model_yaml(model_yaml_path, model_config)
                print(f"已自动生成 model.yaml: {model_yaml_path}")
        if not self.app_zh.text() or not self.app_en.text() or not self.app_tw.text():
            print(lang["app_name_cannot_be_empty"][lang_id])
            #弹出对话框
            QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], lang["app_name_cannot_be_empty"][lang_id])
            return
        if not self.title_zh.text() or not self.title_en.text() or not self.title_tw.text():
            print(lang["title_name_cannot_be_empty"][lang_id])
            #弹出对话框
            QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], lang["title_name_cannot_be_empty"][lang_id])
            return
        
        self.export_btn.setText(lang["converting_please_wait"][lang_id])
        self.export_btn.repaint()   # 强制刷新按钮
        QApplication.processEvents()  # 处理事件队列，刷新界面
        
        
        #分析model.yaml，获取base_model
        model_yaml_path = os.path.join(self.model_dataset_dir, "model.yaml")
        print(f"model_yaml_path={model_yaml_path}",flush=True)
        with open(model_yaml_path, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
            print(f"model_config={model_config}",flush=True)
        self.base_model = model_config.get("base_model", "yolov8n")

        # 分类模型数据集中没有data.yaml,未来兼容性，我们生成一个
        if self.base_model.endswith("-cls"):
            #生成分类模型的data.yaml
            with open(model_yaml_path, "r", encoding="utf-8") as f:
                model_config = yaml.safe_load(f)
                labels = model_config.get("labels", [])
                print(f"labels={labels}",flush=True)
                yaml_path = os.path.join(self.model_dataset_dir, "data.yaml")
                # 生成新的data.yaml,注意分类的图片没有images上层目录
                new_data = {
                    "path": "./",
                    "train": "./train",
                    "names": labels
                }
                with open(os.path.join(self.model_dataset_dir, "data.yaml"), "w", encoding="utf-8") as f:
                    yaml.dump(new_data, f, default_flow_style=False, allow_unicode=True)

        #读取数据集标签
        yaml_path = os.path.join(self.model_dataset_dir, "data.yaml")
        source_config, name_list = get_name_list_from_data_yaml(yaml_path)
        print(f"source_config={source_config}",flush=True)
        #用户自定义目录，我们使用了统一格式，但是cls的data.yaml中train目录可能是./train,这里兼容一下
        if os.path.exists(os.path.join(self.model_dataset_dir, source_config["train"])):
            self.dataset_path = os.path.join(self.model_dataset_dir, source_config["train"])
        else:
            self.dataset_path = os.path.join(self.model_dataset_dir, "images","train")

        conf_data = copy.deepcopy(conf_template)
        conf_data["conf"]["application"] = "dfrobot_" + clean_name(self.app_en.text())
        conf_data["conf"]["model_attach"]["classes"]["zh-CN"] = name_list
        conf_data["conf"]["model_attach"]["classes"]["zh-TW"] = name_list
        conf_data["conf"]["model_attach"]["classes"]["en"] = name_list
        conf_data["conf"]["model_info"][0]["name"] = mindplus_base_model_to_kmodel_base_model[self.base_model][1]
        conf_data["conf"]["model_info"][0]["filename"] = conf_data["conf"]["application"] + ".kmodel"
        conf_data["conf"]["defconfig"]["conf_thres"] = self.threshold_slider.value() / 100

        if self.base_model.endswith("-seg"):
            conf_data["conf"]["defconfig"]["det_thres"] = self.threshold_slider.value() / 100
            conf_data["conf"]["defconfig"]["nms_thres"] = 0.2
            conf_data["conf"]["defconfig"]["msk_thres"] = conf_data["conf"]["defconfig"]["det_thres"]
        elif self.base_model.endswith("-cls"):
            conf_data["conf"]["defconfig"]["rslt_max_num"] = len(name_list)
        else:
            conf_data["conf"]["defconfig"]["det_thres"] = self.threshold_slider.value() / 100
            conf_data["conf"]["defconfig"]["nms_thres"] = 0.2

        shutil.rmtree("model_output")
        os.makedirs("model_output", exist_ok=True)

        with open("model_output/conf.json", "w", encoding="utf-8") as f:
            json.dump(conf_data, f, ensure_ascii=False, indent=4)

        desc_data = copy.deepcopy(desc_template)
        desc_data["desc"]["application_name"] = [
            self.app_en.text().replace("\\n", "\n"), self.app_zh.text().replace("\\n", "\n"), self.app_tw.text().replace("\\n", "\n")
        ]
        desc_data["desc"]["application_title"] = [
            self.title_en.text().replace("\\n", "\n"), self.title_zh.text().replace("\\n", "\n"), self.title_tw.text().replace("\\n", "\n")
        ]

        desc_data["desc"]["base_model"] = mindplus_base_model_to_kmodel_base_model[self.base_model][0]

        with open("model_output/desc.json", "w", encoding="utf-8") as f:
            json.dump(desc_data, f, ensure_ascii=False, indent=4)
        
        icon_file = self._conf["comm"]["icon_file"]
        if os.path.exists(icon_file):
            shutil.copy(icon_file, os.path.join("model_output/icon.png"))
        
        # 创建空文件
        open(f"model_output/app.{conf_data['conf']['application']}", "w").close()

        onnx_path = os.path.join(self.model_dataset_dir, "best.onnx")
        kmodel_path = os.path.join("model_output", conf_data["conf"]["model_info"][0]["filename"])
        output_zip = conf_data["conf"]["application"]

        self.thread = ConvertThread(onnx_path, kmodel_path, self.dataset_path, "kmodel_conf.toml", output_zip)
        self.thread.finished.connect(self.on_conversion_finished)
        self.thread.start()
        print("正在转换")

    def on_conversion_finished(self, file_path):
        self.export_btn.setText(lang["convert_and_package"][lang_id])
        QMessageBox.information(
            self,
            lang["conversion_complete_title"][lang_id],
            lang["conversion_complete_message"][lang_id].format(path=file_path),
        )
        print(f"转换完成！文件路径: {file_path}")        

    def pack(self):
        # 打包 ZIP
        conf_data = copy.deepcopy(conf_template)
        conf_data["conf"]["application"] = "dfrobot_" + clean_name(self.app_en.text())
        zip_with_md5(base_name=conf_data["conf"]["application"])
        print("转换完成！")

if __name__ == "__main__":
    language_code = locale.getdefaultlocale()[0]
    print(f"默认语言环境: {language_code}")
    if language_code.startswith("zh_CN"):
        lang_id = 1
    else:
        lang_id = 0
    os.makedirs("model_output", exist_ok=True)
    os.makedirs("model_input", exist_ok=True)
    app = QApplication(sys.argv)
    window = ModelExportApp()
    window.show()
    sys.exit(app.exec_())
