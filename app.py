# -*- coding: utf-8 -*-

import os
import sys
import zipfile
import hashlib
import shutil
import yaml
import json
import re
import toml
import convertor
from PIL import Image
from PyQt5.QtWidgets import (
    QFrame, QApplication, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit,
    QFileDialog, QComboBox, QSlider, QHBoxLayout, QVBoxLayout, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

conf_template = {
    "conf": {
        "application": "",
        "defconfig": {"det_thres": 0.3, "nms_thres": 0.6},
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
        "stream": True,
        "version": "0.1"
    }
}

def clean_name(name):
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

class ModelExportApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mindplus模型转二哈安装包")
        self.resize(800, 600)
        with open("app_conf.toml", 'r', encoding='utf-8') as f:
            self._conf = toml.load(f)
            print(self._conf)
        self.zip_file = None
        self.icon_file = self._conf["comm"]["icon_file"]
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout()

        # 模型包选择
        self.zip_model_button = QPushButton("选择模型包 (*.zip)")
        self.zip_model_button.clicked.connect(self.select_zip)
        self.zip_model_label = QLineEdit()
        self.zip_model_label.setReadOnly(True)
        layout.addWidget(self.zip_model_button, 0, 0)
        layout.addWidget(self.zip_model_label, 0, 1)

        # 数据包选择
        self.zip_dataset_button = QPushButton("选择模型包 (*.zip)")
        self.zip_dataset_button.clicked.connect(self.select_zip)
        self.zip_dataset_label = QLineEdit()
        self.zip_dataset_label.setReadOnly(True)
        layout.addWidget(self.zip_dataset_button, 1, 0)
        layout.addWidget(self.zip_dataset_label, 1, 1)

        # 图标选择
        self.icon_button = QPushButton("选择图标")
        self.icon_button.clicked.connect(self.select_icon)
        self.icon_preview = QLabel()
        if self.icon_file and os.path.exists(self.icon_file):
            pixmap = QPixmap(self.icon_file)
            pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.icon_preview.setPixmap(pixmap)
        layout.addWidget(self.icon_button, 2, 0)
        layout.addWidget(self.icon_preview, 2, 1)

        # 模型选择
        layout.addWidget(QLabel("选择模型"), 3, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n"])
        layout.addWidget(self.model_combo, 3, 1)

        # 分割线：应用名称
        line_app = QFrame()
        line_app.setFrameShape(QFrame.HLine)
        line_app.setFrameShadow(QFrame.Sunken)
        layout.addWidget(QLabel("应用名称设置"), 4, 0, 1, 2)
        layout.addWidget(line_app, 5, 0, 1, 2)

        # 应用名称输入
        layout.addWidget(QLabel("简体中文(必填)"), 6, 0)
        self.app_zh = QLineEdit(self._conf["comm"]["app_name_zh_CN"])
        layout.addWidget(self.app_zh, 6, 1)

        layout.addWidget(QLabel("繁体中文(必填)"), 7, 0)
        self.app_tw = QLineEdit(self._conf["comm"]["app_name_zh_TW"])
        layout.addWidget(self.app_tw, 7, 1)

        layout.addWidget(QLabel("English(必填)"), 8, 0)
        self.app_en = QLineEdit(self._conf["comm"]["app_name_EN"])
        layout.addWidget(self.app_en, 8, 1)

        # 分割线：标题
        line_title = QFrame()
        line_title.setFrameShape(QFrame.HLine)
        line_title.setFrameShadow(QFrame.Sunken)
        layout.addWidget(QLabel("标题设置"), 9, 0, 1, 2)
        layout.addWidget(line_title, 10, 0, 1, 2)

        # 标题输入
        layout.addWidget(QLabel("标题简体中文"), 11, 0)
        self.title_zh = QLineEdit("细胞识别")
        layout.addWidget(self.title_zh, 11, 1)

        layout.addWidget(QLabel("标题繁体中文"), 12, 0)
        self.title_tw = QLineEdit("細胞識別")
        layout.addWidget(self.title_tw, 12, 1)

        layout.addWidget(QLabel("标题English"), 13, 0)
        self.title_en = QLineEdit("Cell Recognition")
        layout.addWidget(self.title_en, 13, 1)

        # 默认识别阈值 (滑条 + 数值)
        layout.addWidget(QLabel("默认识别阈值(0-1)"), 14, 0)
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(30)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("0.30")
        threshold_layout.addWidget(self.threshold_label)
        layout.addLayout(threshold_layout, 14, 1)

        bottom_btn_layout = QHBoxLayout()
        bottom_btn_layout.addStretch(1)
        # 保存配置
        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self.save_conf)
        bottom_btn_layout.addWidget(self.save_btn, 1)

        bottom_btn_layout.addStretch(1)
        
        # 转换按钮
        self.export_btn = QPushButton("转换")
        self.export_btn.clicked.connect(self.export_model)
        bottom_btn_layout.addWidget(self.export_btn, 1)

        bottom_btn_layout.addStretch(1)
        layout.addLayout(bottom_btn_layout, 15, 0, 1, 2)

        self.setLayout(layout)


    def update_threshold_label(self, value):
        # 转换为 0.00 ~ 1.00
        f_value = value / 100.0
        self.threshold_label.setText(f"{f_value:.2f}")

    def select_zip(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择ZIP文件", "", "ZIP files (*.zip)")
        if file:
            self.zip_file = file
            self.zip_label.setText(file)

    def select_icon(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择图标", "", "PNG files (*.png)")
        if file:
            self.icon_file = file
            img = Image.open(file)
            img = img.resize((60, 60))
            os.makedirs("model_output", exist_ok=True)
            img.save("model_output/icon.png")
            pixmap = QPixmap("model_output/icon.png")
            self.icon_preview.setPixmap(pixmap)

    def save_conf(self):
        self._conf["comm"]["icon_file"] = self.icon_file
        self._conf["comm"]["app_name_zh_CN"] = self.app_zh.text()
        self._conf["comm"]["app_name_EN"] = self.app_en.text()
        self._conf["comm"]["app_name_zh_TW"] = self.app_tw.text()

        self._conf["comm"]["title_name_zh_CN"] = self.title_zh.text()
        self._conf["comm"]["title_name_EN"] = self.title_en.text()
        self._conf["comm"]["title_name_zh_TW"] = self.title_tw.text()

        with open("app_conf.toml", 'w', encoding='utf-8') as f:
            toml.dump(self._conf, f)

    def export_model(self):
        if not self.zip_file:
            print("请选择ZIP文件")
            return
        # 解压
        self.export_btn.setText("转换中......")
        extract_zip(self.zip_file, "model_input")
        yaml_path = os.path.join("model_input", "dataset", "data.yaml")
        with open(yaml_path, "r", encoding="utf-8") as f:
            source_config = yaml.safe_load(f)
        names = source_config.get("names", {})
        name_list = [names[i] for i in sorted(names.keys())]

        conf_data = conf_template
        conf_data["conf"]["application"] = "dfrobot_" + clean_name(self.app_en.text())
        conf_data["conf"]["model_attach"]["classes"]["zh-CN"] = name_list
        conf_data["conf"]["model_attach"]["classes"]["zh-TW"] = name_list
        conf_data["conf"]["model_attach"]["classes"]["en"] = name_list
        conf_data["conf"]["model_info"][0]["filename"] = conf_data["conf"]["application"] + ".kmodel"
        conf_data["conf"]["defconfig"]["det_thres"] = self.threshold_slider.value() / 100

        os.makedirs("model_output", exist_ok=True)
        with open("model_output/conf.json", "w", encoding="utf-8") as f:
            json.dump(conf_data, f, ensure_ascii=False, indent=4)

        desc_data = desc_template
        desc_data["desc"]["application_name"] = [
            self.app_en.text(), self.app_zh.text(), self.app_tw.text()
        ]
        desc_data["desc"]["application_title"] = [
            self.title_en.text(), self.title_zh.text(), self.title_tw.text()
        ]
        with open("model_output/desc.json", "w", encoding="utf-8") as f:
            json.dump(desc_data, f, ensure_ascii=False, indent=4)

        # 创建空文件
        open(f"model_output/app.{conf_data['conf']['application']}", "w").close()
        convertor.make("model_input/model/best.onnx","model_output/"+conf_data["conf"]["model_info"][0]["filename"],"kmodel_conf.toml")
        # 打包 ZIP
        zip_with_md5(base_name=conf_data["conf"]["application"])
        self.export_btn.setText("转换")
        print("转换完成！")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelExportApp()
    window.show()
    sys.exit(app.exec_())
