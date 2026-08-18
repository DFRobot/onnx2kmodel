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
    QApplication, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox,
    QFileDialog, QSlider, QHBoxLayout, QVBoxLayout, QMessageBox, QGroupBox,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QEvent, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QProcess

# 优先使用当前目录下的 qdarkstyle_dfrobot，避免依赖外部安装的包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#import qdarkstyle_dfrobot as qdarkstyle
import qdarkstyle

import io

# Debug 开关：True 时显示“保存配置”和“仅打包”按钮，False 时隐藏
DEBUG = True

# PyInstaller 打包后资源/配置文件放在 exe 同目录；python 运行时为脚本所在目录
BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))

# PyInstaller 无控制台（windowed）模式下 sys.stdout 为 None，需要跳过
if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

lang_id = 0
lang = {
    "name": ["name", "名字", "名字", "nom", "이름", "nombre", "nome", "名前"],
    "data": ["Model ", "数据包", "數據包", "Modèle ", "모델 데이터 ", "Modelo ", "Pacote de modelo ", "モデル データ "],
    "icon": ["Icon", "图标", "圖標", "Icône", "아이콘", "Icono", "Ícone", "アイコン"],
    "custom": ["Custom", "自定义", "自定義", "Personnalisé", "사용자 정의", "Personalizado", "Personalizado", "カスタム"],
    "select_mode": ["Data Source", "数据源", "數據源", "Source de données", "데이터 소스", "Fuente de datos", "Fonte de dados", "データソース"],
    "input_source": ["Input Source", "输入数据源", "輸入數據源", "Source de données", "입력 소스", "Fuente de entrada", "Fonte de entrada", "入力ソース"],
    "custom_directory": ["Custom Directory", "用户自定义目录", "用戶自定義目錄", "Répertoire personnalisé", "사용자 지정 디렉터리", "Directorio personalizado", "Diretório personalizado", "カスタムディレクトリ"],
    "select_model_package": ["Select Model Package", "选择模型包", "選擇模型包", "Sélectionner le paquet de modèle", "모델 패키지 선택", "Seleccionar paquete de modelo", "Selecionar pacote de modelo", "モデルパッケージ選択"],
    "select_dataset_package": ["Select Dataset Package", "选择数据集包", "選擇數據集包", "Sélectionner le paquet de jeu de données", "데이터세트 패키지 선택", "Seleccionar paquete de conjunto de datos", "Selecionar pacote de conjunto de dados", "データセットパッケージ選択"],
    "select_icon": ["Select Icon", "选择图标", "選擇圖標", "Sélectionner l'icône", "아이콘 선택", "Seleccionar icono", "Selecionar ícone", "アイコン選択"],
    "app_name": ["App Name", "应用名称", "應用名稱", "Nom de l'application", "앱 이름", "Nombre de la aplicación", "Nome do aplicativo", "アプリ名"],
    "add_other_language": ["Add other language", "添加其他语言", "添加其他語言", "Ajouter une langue", "다른 언어 추가", "Agregar otro idioma", "Adicionar outro idioma", "他の言語を追加"],
    "delete": ["Delete", "删除", "刪除", "Supprimer", "삭제", "Eliminar", "Excluir", "削除"],
    "app_name_placeholder": [
        "Set your app name here. Use 「Enter」 to wrap.",
        "在此设置您的应用名称。使用「Enter」换行。",
        "在此設定您的應用程式名稱。使用「Enter」換行。",
        "Saisissez le nom de votre application ici. Utilisez 「Enter」 pour passer à la ligne.",
        "여기에 앱 이름을 입력하세요. 줄바꿈하려면 「Enter」를 사용하세요.",
        "Introduce aquí el nombre de tu aplicación. Usa 「Enter」 para saltar de línea.",
        "Insira o nome do seu aplicativo aqui. Use 「Enter」 para quebrar a linha.",
        "ここにアプリ名を入力してください。「Enter」を押して改行します。"
    ],
    "simplified_chinese": ["Simplified Chinese", "简体中文", "簡體中文", "Chinois simplifié", "간체 중국어", "Chino simplificado", "Chinês simplificado", "簡体中国語"],
    "traditional_chinese": ["Traditional Chinese", "繁体中文", "繁體中文", "Chinois traditionnel", "번체 중국어", "Chino tradicional", "Chinês tradicional", "繁体中国語"],
    "title_settings": ["Title Settings", "标题名称", "標題名稱", "Paramètres du titre", "제목 설정", "Configuración del título", "Configurações do título", "タイトル設定"],
    "detection_threshold": ["Detection Threshold", "识别阈值", "識別閾值", "Seuil de détection", "인식 임계값", "Umbral de detección", "Limite de detecção", "認識しきい値"],
    "save_config": ["Save Config", "保存配置", "保存配置", "Enregistrer config", "설정 저장", "Guardar configuración", "Salvar configuração", "設定を保存"],
    "convert_and_package": ["Convert and Package", "转换&打包", "轉換&打包", "Convertir et packager", "변환 및 패키징", "Convertir y empaquetar", "Converter e empacotar", "変換とパッケージ"],
    "pack_only": ["Pack Only", "仅打包", "僅打包", "Packager seul", "패키징만", "Solo empaquetar", "Apenas empacotar", "パッケージのみ"],
    "app_title": ["Mindplus Model to K230 Installer", "Mindplus模型转二哈安装包", "Mindplus模型轉二哈安裝包", "Installateur Mindplus vers K230", "Mindplus 모델 K230 설치 패키지", "Instalador Mindplus a K230", "Instalador Mindplus para K230", "Mindplus モデル K230 インストーラー"],
    "select_custom_directory": ["Select Custom Directory", "选择用户自定义目录", "選擇用戶自定義目錄", "Sélectionner le répertoire personnalisé", "사용자 지정 디렉터리 선택", "Seleccionar directorio personalizado", "Selecionar diretório personalizado", "カスタムディレクトリ選択"],
    "select_zip_file": ["Select ZIP File", "选择ZIP文件", "選擇ZIP文件", "Sélectionner le fichier ZIP", "ZIP 파일 선택", "Seleccionar archivo ZIP", "Selecionar arquivo ZIP", "ZIPファイル選択"],
    "app_name_cannot_be_empty": ["App Name cannot be empty", "应用名称不能为空", "應用名稱不能為空", "Le nom de l'application ne peut pas être vide", "앱 이름은 비워둘 수 없습니다", "El nombre de la aplicación no puede estar vacío", "O nome do aplicativo não pode estar vazio", "アプリ名を空にできません"],
    "title_name_cannot_be_empty": ["Title Name cannot be empty", "标题名称不能为空", "標題名稱不能為空", "Le titre ne peut pas être vide", "제목은 비워둘 수 없습니다", "El título no puede estar vacío", "O título não pode estar vazio", "タイトルを空にできません"],
    "converting_please_wait": ["Converting, please wait...", "转换中......, 需要几分钟，请耐心等待", "轉換中......，需要幾分鐘，請耐心等待", "Conversion en cours, veuillez patienter...", "변환 중... 잠시만 기다려 주세요", "Convirtiendo, espere...", "Convertendo, aguarde...", "変換中です。しばらくお待ちください..."],
    "dialog_warning_title": ["Warning", "警告", "警告", "Avertissement", "경고", "Advertencia", "Aviso", "警告"],
    "model_zip_not_found": ["Model package not found", "模型包不存在", "模型包不存在", "Paquet modèle introuvable", "모델 패키지를 찾을 수 없습니다", "Paquete de modelo no encontrado", "Pacote de modelo não encontrado", "モデルパッケージが見つかりません"],
    "dataset_zip_not_found": ["Dataset package not found", "数据集包不存在", "數據集包不存在", "Paquet de données introuvable", "데이터세트 패키지를 찾을 수 없습니다", "Paquete de datos no encontrado", "Pacote de dados não encontrado", "データセットパッケージが見つかりません"],
    "user_dir_not_found": ["Custom directory not found", "用户自定义目录不存在", "用戶自定義目錄不存在", "Répertoire personnalisé introuvable", "사용자 지정 디렉터리를 찾을 수 없습니다", "Directorio personalizado no encontrado", "Diretório personalizado não encontrado", "カスタムディレクトリが見つかりません"],
    "onnx_not_found": ["best.onnx not found", "best.onnx 不存在", "best.onnx 不存在", "best.onnx introuvable", "best.onnx를 찾을 수 없습니다", "best.onnx no encontrado", "best.onnx não encontrado", "best.onnxが見つかりません"],
    "model_output_not_found": ["Output directory not ready, please convert first", "输出目录未就绪，请先转换", "輸出目錄未就緒，請先轉換", "Répertoire de sortie non prêt, veuillez d'abord convertir", "출력 디렉터리가 준비되지 않았습니다. 먼저 변환하세요", "Directorio de salida no listo, primero convierta", "Diretório de saída não pronto, converta primeiro", "出力ディレクトリの準備ができていません。先に変換してください"],
    "pack_failed": ["Pack failed", "打包失败", "打包失敗", "Échec du pack", "패키징 실패", "Error al empaquetar", "Falha ao empacotar", "パッケージングに失敗しました"],
    "pack_complete_title": ["Pack Complete", "打包完成", "打包完成", "Pack terminé", "패키징 완료", "Empaquetado completado", "Empacotamento concluído", "パッケージング完了"],
    "pack_complete_message": [
        "Pack complete!\n{path}",
        "打包完成！\n{path}",
        "打包完成！\n{path}",
        "Pack terminé !\n{path}",
        "패키징 완료!\n{path}",
        "¡Empaquetado completado!\n{path}",
        "Empacotamento concluído!\n{path}",
        "パッケージング完了！\n{path}",
    ],
    "conversion_complete_title": ["Conversion Complete", "转换完成", "轉換完成", "Conversion terminée", "변환 완료", "Conversión completada", "Conversão concluída", "変換完了"],
    "conversion_complete_message": [
        "Conversion complete!\n{path}",
        "转换完成！\n{path}",
        "轉換完成！\n{path}",
        "Conversion terminée !\n{path}",
        "변환 완료!\n{path}",
        "¡Conversión completada!\n{path}",
        "Conversão concluída!\n{path}",
        "変換完了！\n{path}",
    ],
    "conversion_failed": ["Conversion Failed", "转换失败", "轉換失敗", "Échec de la conversion", "변환 실패", "Conversión fallida", "Falha na conversão", "変換失敗"],
    "english": ["English", "英文", "英文", "Anglais", "영어", "Inglés", "Inglês", "英語"],
    "french": ["French", "法语", "法語", "Français", "프랑스어", "Francés", "Francês", "フランス語"],
    "korean": ["Korean", "韩语", "韓語", "Coréen", "한국어", "Coreano", "Coreano", "韓国語"],
    "spanish": ["Spanish", "西班牙语", "西班牙語", "Espagnol", "스페인어", "Español", "Espanhol", "スペイン"],
    "portuguese": ["Portuguese (Brazil)", "巴西葡语", "巴西葡語", "Portugais (Brésil)", "포르투갈어(브라질)", "Portugués (Brasil)", "Português (Brasil)", "ポルトガル語 (ブラジル)"],
    "japanese": ["Japanese", "日语", "日語", "Japonais", "일본어", "Japonés", "Japonês", "日本語"],
}

LANG_KEYS = ["en", "zh-CN", "zh-TW", "fr", "ko", "es", "pt-BR", "ja"]
LANG_DISPLAY_NAMES = ["English", "简体中文", "繁体中文", "Français", "한국어", "Español", "Português (Brasil)", "日本語"]

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
        "application_name": {
            "en": "name",
            "zh-CN": "名字",
            "zh-TW": "名字",
            "fr": "nom",
            "ko": "이름",
            "es": "nombre",
            "pt-BR": "nome",
            "ja": "名前"
        },
        "application_title": {
            "en": "title",
            "zh-CN": "抬头",
            "zh-TW": "抬頭",
            "fr": "titre",
            "ko": "제목",
            "es": "título",
            "pt-BR": "título",
            "ja": "タイトル"
        },
        "base_model": "yolov8n-det",
        "stream": True,
        "version": "0.2"
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
    # 先去除首尾空白，再将中间空格、Tab、换行及 \n 转义序列替换为下划线
    return name.strip().replace(" ", "_").replace("\t", "_").replace("\n", "_").replace("\\n", "_")

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


class ModelExportApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(lang["app_title"][lang_id])
        self.resize(800, 545)
        with open(os.path.join(BASE_DIR, "app_conf.toml"), 'r', encoding='utf-8') as f:
            self._conf = tomlkit.parse(f.read())
            print(self._conf)
        self._normalize_app_name_newlines()
        self._ensure_app_name_keys()
        self.init_ui()

    def _normalize_app_name_newlines(self):
        """将配置中应用名称的 \\n 转义序列转换为真实换行符，便于 QTextEdit 换行显示。"""
        comm = self._conf.get("comm", {})
        # 兼容旧格式：将 app_name_A / app_name_by_lang 转换为 app_name_<key>
        if "app_name_A" in comm or "app_name_by_lang" in comm:
            app_name_a = comm.get("app_name_A", "")
            app_name_by_lang = comm.get("app_name_by_lang", {})
            if isinstance(app_name_by_lang, dict):
                # app_name_A 是系统语言对应的值，优先放入系统语言 key
                system_key = self._get_system_lang_key()
                if f"app_name_{system_key}" not in comm and app_name_a:
                    comm[f"app_name_{system_key}"] = app_name_a
                # 合并 app_name_by_lang 中已有的语言
                for key, value in app_name_by_lang.items():
                    if key in LANG_KEYS and value:
                        comm[f"app_name_{key}"] = value
                # 缺失的语言用系统语言补齐
                system_value = comm.get(f"app_name_{system_key}", app_name_a)
                for key in LANG_KEYS:
                    if f"app_name_{key}" not in comm or not comm[f"app_name_{key}"]:
                        comm[f"app_name_{key}"] = system_value
        for key in LANG_KEYS:
            name_key = f"app_name_{key}"
            if name_key in comm and isinstance(comm[name_key], str):
                comm[name_key] = comm[name_key].replace("\\n", "\n")
            title_key = f"title_name_{key}"
            if title_key in comm and isinstance(comm[title_key], str):
                comm[title_key] = comm[title_key].replace("\\n", "\n")

    def _get_system_lang_key(self):
        """获取系统语言对应的 LANG_KEYS 键。"""
        return LANG_KEYS[lang_id]

    def _ensure_app_name_keys(self):
        """确保 comm 中所有 app_name_<key> 和 title_name_<key> 都有值，缺失时以系统语言补齐。"""
        comm = self._conf["comm"]
        system_key = self._get_system_lang_key()
        system_name = comm.get(f"app_name_{system_key}", "")
        system_title = comm.get(f"title_name_{system_key}", system_name.replace("\n", " "))
        for key in LANG_KEYS:
            name_key = f"app_name_{key}"
            if name_key not in comm or not comm[name_key]:
                comm[name_key] = system_name
            title_key = f"title_name_{key}"
            if title_key not in comm or not comm[title_key]:
                comm[title_key] = system_title

    def closeEvent(self, event):
        """窗口关闭时确保所有定时器和事件处理完毕，避免进程残留。"""
        # 停止所有未完成的 QTimer
        for child in self.findChildren(QTimer):
            if child.isActive():
                child.stop()
        # 确保事件循环退出
        QApplication.instance().quit()
        super().closeEvent(event)

    def _adjust_height_to_content(self):
        """根据当前布局高度调整窗口高度，保持宽度不变。"""
        self.layout().update()
        self.resize(self.width(), self.sizeHint().height())

    def init_ui(self):
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(15, 25, 15, 15)

        # --- 语言 与 图标 并排 ---
        language_icon_layout = QHBoxLayout()

        # --- 语言选择 ---
        self.language_group = QGroupBox("Language")
        self.language_group.setMinimumWidth(260)
        self.language_group.setFixedHeight(102)
        language_layout = QHBoxLayout()
        self.lang_combo = QComboBox()
        self.lang_combo.setMinimumHeight(30)
        self.lang_combo.addItems(["English", "简体中文", "繁体中文", "Français", "한국어", "Español", "Português (Brasil)", "日本語"])
        self.lang_combo.setCurrentIndex(lang_id)
        self.lang_combo.currentIndexChanged.connect(self.lang_changed)
        language_layout.addWidget(self.lang_combo)
        self.language_group.setLayout(language_layout)
        language_icon_layout.addWidget(self.language_group)

        # --- 图标选择 ---
        self.icon_group = QGroupBox(lang["icon"][lang_id])
        self.icon_group.setFixedHeight(102)
        icon_layout = QHBoxLayout()
        self.icon_button = QPushButton(lang["select_icon"][lang_id])
        self.icon_button.clicked.connect(self.select_icon)
        self.icon_button.setFixedSize(150, 36)
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
        self.icon_group.setLayout(icon_layout)
        language_icon_layout.addWidget(self.icon_group, 1)

        self.main_layout.addLayout(language_icon_layout)
        self.main_layout.addSpacing(10)

        # --- 模式 / 输入数据源 ---
        self.select_mode_group = QGroupBox(lang["select_mode"][lang_id])
        select_mode_layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        self.select_mode_label = QLabel(lang["input_source"][lang_id])
        mode_layout.addWidget(self.select_mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumHeight(30)
        self.mode_combo.addItems(["MindPlus", lang["custom"][lang_id]])
        if self._conf["comm"]["mode"] == "MindPlus":
            self.mode_combo.setCurrentIndex(0)
        else:
            self.mode_combo.setCurrentIndex(1)
        self.mode_combo.currentIndexChanged.connect(self.mode_changed)
        mode_layout.addWidget(self.mode_combo)
        select_mode_layout.addLayout(mode_layout)

        self.select_mode_group.setLayout(select_mode_layout)

        # --- 模型包选择 ---
        model_layout = QHBoxLayout()
        self.zip_model_button = QPushButton(lang["select_model_package"][lang_id] + " (*.zip)")
        self.zip_model_button.setMinimumHeight(36)
        self.zip_model_button.setFixedWidth(150)
        self.zip_model_button.clicked.connect(lambda: self.select_zip("model"))
        self.zip_model_label = QLineEdit(self._conf["mindplus_options"]["model_zip"])
        self.zip_model_label.setReadOnly(True)
        model_layout.addWidget(self.zip_model_button)
        model_layout.addSpacing(10)
        model_layout.addWidget(self.zip_model_label)
        select_mode_layout.addLayout(model_layout)

        # --- 数据包选择 ---
        dataset_layout = QHBoxLayout()
        self.zip_dataset_button = QPushButton(lang["select_dataset_package"][lang_id] + " (*.zip)")
        self.zip_dataset_button.setMinimumHeight(36)
        self.zip_dataset_button.setFixedWidth(150)
        self.zip_dataset_button.clicked.connect(lambda: self.select_zip("dataset"))
        self.zip_dataset_label = QLineEdit(self._conf["mindplus_options"]["dataset_zip"])
        self.zip_dataset_label.setReadOnly(True)
        dataset_layout.addWidget(self.zip_dataset_button)
        dataset_layout.addSpacing(10)
        dataset_layout.addWidget(self.zip_dataset_label)
        select_mode_layout.addLayout(dataset_layout)

        # --- 自定义目录选择 ---
        user_layout = QHBoxLayout()
        self.user_dir_button = QPushButton(lang["custom_directory"][lang_id])
        self.user_dir_button.setMinimumSize(120, 36)
        self.user_dir_button.clicked.connect(self.select_user_dir)
        self.user_dir_label = QLineEdit(self._conf["user_options"]["user_dir"])
        self.user_dir_label.setReadOnly(True)
        user_layout.addWidget(self.user_dir_button)
        user_layout.addSpacing(10)
        user_layout.addWidget(self.user_dir_label)
        select_mode_layout.addLayout(user_layout)

        self.main_layout.addWidget(self.select_mode_group)
        self.main_layout.addSpacing(10)

        # --- 应用名称 ---
        self.app_name_group = QGroupBox(lang["app_name"][lang_id])
        self.app_name_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        app_name_layout = QVBoxLayout()
        app_name_layout.setSpacing(5)
        app_name_layout.setContentsMargins(8, 8, 8, 8)

        # 当前界面语言的应用名称（与额外语言行同一行，减少纵向空白）
        current_lang_row = QHBoxLayout()
        self.app_name_current_lang_label = QLabel(LANG_DISPLAY_NAMES[lang_id])
        self.app_name_current_lang_label.setStyleSheet("font-weight: bold;")
        # 固定标签宽度，与额外语言行左对齐
        fm = self.app_name_current_lang_label.fontMetrics()
        label_width = max(fm.boundingRect(name).width() for name in LANG_DISPLAY_NAMES) + 40
        self.app_name_current_lang_label.setFixedWidth(label_width)
        current_lang_row.addWidget(self.app_name_current_lang_label)
        current_lang_row.addSpacing(5)
        current_key = LANG_KEYS[lang_id]
        app_name_text = self._conf["comm"].get(f"app_name_{current_key}", "")
        self.app_name_A = QTextEdit()
        self.app_name_A.setPlainText(app_name_text)
        self._set_text_edit_two_lines(self.app_name_A)
        self.app_name_A.setPlaceholderText(lang["app_name_placeholder"][lang_id])
        self.app_name_A.textChanged.connect(self._update_app_name_extra_placeholders)
        self.app_name_A.installEventFilter(self)
        self.app_name_A.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        current_lang_row.addWidget(self.app_name_A, 1)
        app_name_layout.addLayout(current_lang_row)

        # 动态添加的其他语言行
        self.app_name_extra_rows_layout = QVBoxLayout()
        self.app_name_extra_rows_layout.setSpacing(5)
        app_name_layout.addLayout(self.app_name_extra_rows_layout)

        # 添加其他语言
        add_lang_layout = QHBoxLayout()
        add_lang_layout.addStretch(1)
        self.app_name_lang_combo = QComboBox()
        self.app_name_lang_combo.setMinimumHeight(30)
        self.add_app_name_lang_btn = QPushButton(lang["add_other_language"][lang_id])
        self.add_app_name_lang_btn.setMinimumHeight(24)
        self.add_app_name_lang_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._set_button_auto_width(self.add_app_name_lang_btn)
        self.add_app_name_lang_btn.clicked.connect(self.add_app_name_lang)
        add_lang_layout.addWidget(self.app_name_lang_combo)
        add_lang_layout.addWidget(self.add_app_name_lang_btn)
        app_name_layout.addLayout(add_lang_layout)

        self.app_name_group.setLayout(app_name_layout)
        self.main_layout.addWidget(self.app_name_group)
        self.main_layout.addSpacing(10)

        # 多语言缓存，key 为 LANG_KEYS 中的语言代码，从 app_name_<key> 读取
        self.app_name_by_lang = {}
        for key in LANG_KEYS:
            value = self._conf["comm"].get(f"app_name_{key}", "")
            if value:
                self.app_name_by_lang[key] = value
        # 用户手动添加的语言列表（系统语言总是显示，不计入）
        self.user_added_langs = self._conf["comm"].get("user_added_langs", [])
        if not isinstance(self.user_added_langs, list):
            self.user_added_langs = []
        self._current_app_name_lang_id = lang_id
        self.app_name_extra_rows = {}  # lang_key -> widgets dict
        self.extra_lang_keys = set()    # 额外语言集合

        # 初始化额外语言行和下拉框
        self._sync_from_config()
        self._refresh_lang_combos()

        # --- 阈值设置 ---
        self.threshold_group = QGroupBox(lang["detection_threshold"][lang_id])
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(self._conf["comm"]["det_threshold"]*100))
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        self.threshold_label = QLabel(str(self._conf["comm"]["det_threshold"]))
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        self.threshold_group.setLayout(threshold_layout)
        self.main_layout.addWidget(self.threshold_group)
        self.main_layout.addSpacing(10)

        # --- 底部按钮 ---
        btn_layout = QHBoxLayout()
        self.export_btn = QPushButton(lang["convert_and_package"][lang_id])
        self.export_btn.clicked.connect(self.export_model)
        self.export_btn.setMinimumSize(100, 36)

        btn_layout.addStretch(1)
        if DEBUG:
            self.save_btn = QPushButton(lang["save_config"][lang_id])
            self.save_btn.clicked.connect(self.save_conf)
            self.save_btn.setMinimumSize(100, 36)
            btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.export_btn)
        if DEBUG:
            self.pack_btn = QPushButton(lang["pack_only"][lang_id])
            self.pack_btn.clicked.connect(self.pack)
            self.pack_btn.setMinimumSize(100, 36)
            btn_layout.addWidget(self.pack_btn)
        btn_layout.addStretch(1)
        self.main_layout.addLayout(btn_layout)

        # 设置主布局
        self.setLayout(self.main_layout)

        # 初始化控件显隐
        self.mode_changed(self.mode_combo.currentIndex())

    def update_threshold_label(self, value):
        # 转换为 0.00 ~ 1.00
        f_value = value / 100.0
        self._conf["comm"]["det_threshold"] = float("{:.2f}".format(f_value))
        self.threshold_label.setText(f"{f_value:.2f}")

    def select_zip(self,file_type):
        file, _ = QFileDialog.getOpenFileName(self, lang["select_zip_file"][lang_id], "", "ZIP files (*.zip)",
                                              options=QFileDialog.DontUseNativeDialog)
        if file:
            if file_type == "model":
                self._conf["mindplus_options"]["model_zip"] = file
                self.zip_model_label.setText(file)
            elif file_type == "dataset":
                self._conf["mindplus_options"]["dataset_zip"] = file
                self.zip_dataset_label.setText(file)

    def select_user_dir(self):
        directory = QFileDialog.getExistingDirectory(self, lang["select_custom_directory"][lang_id], "",
                                                     options=QFileDialog.DontUseNativeDialog)
        if directory:
            self._conf["user_options"]["user_dir"] = directory
            self.user_dir_label.setText(directory)

    def select_icon(self):
        file, _ = QFileDialog.getOpenFileName(self, lang["select_icon"][lang_id], "", "PNG files (*.png)",
                                              options=QFileDialog.DontUseNativeDialog)
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

    def _update_app_name_placeholder(self):
        """根据当前界面语言设置应用名称输入框的占位提示文字。"""
        self.app_name_A.setPlaceholderText(lang["app_name_placeholder"][lang_id])

    def _update_app_name_extra_placeholders(self):
        """其他语言应用名称为空时，显示默认语言输入作为灰色占位；
        默认输入为空时，显示当前语言的 app_name_placeholder。"""
        default_text = self.app_name_A.toPlainText()
        for lang_key, widgets in self.app_name_extra_rows.items():
            lang_index = LANG_KEYS.index(lang_key)
            if default_text:
                widgets["line_edit"].setPlaceholderText(default_text)
            else:
                widgets["line_edit"].setPlaceholderText(lang["app_name_placeholder"][lang_index])

    def _set_button_auto_width(self, button, padding=24):
        """根据按钮文本长度设置最小宽度，保持左右留白一致。"""
        fm = button.fontMetrics()
        text_width = fm.horizontalAdvance(button.text())
        button.setMinimumWidth(text_width + padding)

    def _set_text_edit_two_lines(self, text_edit):
        """根据当前字体设置 QTextEdit 高度为两行，并隐藏滚动条。"""
        fm = text_edit.fontMetrics()
        line_height = fm.lineSpacing()
        # 框架上下边距大约各占一行间距的 25%，再额外增加 5 像素
        text_edit.setFixedHeight(int(line_height * 2 + line_height * 0.5 + 5))
        text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def eventFilter(self, obj, event):
        """当文本框已有两行时，禁止回车换行。"""
        if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if obj is self.app_name_A or obj in [w["line_edit"] for w in self.app_name_extra_rows.values()]:
                text = obj.toPlainText()
                if text.count("\n") >= 1:
                    return True
        return super().eventFilter(obj, event)

    def _sync_from_config(self):
        """根据配置初始化额外语言输入行，只加载用户手动添加的语言。"""
        # 清空已有行
        for key in list(self.extra_lang_keys):
            self._remove_lang_row(key)
        # 只加载用户手动添加的语言（排除当前系统语言）
        current_key = LANG_KEYS[self._current_app_name_lang_id]
        for key in self.user_added_langs:
            if key == current_key:
                continue
            if key in LANG_KEYS:
                self._add_lang_row(key, self.app_name_by_lang.get(key, ""))

    def _refresh_lang_combos(self):
        """下拉框只显示尚未添加的语言。"""
        current_key = LANG_KEYS[self._current_app_name_lang_id]
        added_keys = self.extra_lang_keys | {current_key}
        available = [(i, name) for i, name in enumerate(LANG_DISPLAY_NAMES) if LANG_KEYS[i] not in added_keys]
        self.app_name_lang_combo.clear()
        if available:
            for i, name in available:
                self.app_name_lang_combo.addItem(name, LANG_KEYS[i])
            self.app_name_lang_combo.setCurrentIndex(0)
            self.add_app_name_lang_btn.setEnabled(True)
        else:
            self.add_app_name_lang_btn.setEnabled(False)

    def _add_lang_row(self, lang_key, app_text=""):
        """添加应用名称的某一语言输入行。"""
        if lang_key in self.extra_lang_keys:
            return
        self.extra_lang_keys.add(lang_key)
        lang_index = LANG_KEYS.index(lang_key)
        lang_name = LANG_DISPLAY_NAMES[lang_index]

        # 应用名称行
        app_row_layout = QHBoxLayout()
        app_label = QLabel(lang_name)
        app_label.setStyleSheet("font-weight: bold;")
        # 固定标签宽度，让所有语言文本框左对齐
        fm = app_label.fontMetrics()
        label_width = max(fm.boundingRect(name).width() for name in LANG_DISPLAY_NAMES) + 40
        app_label.setFixedWidth(label_width)
        app_line_edit = QTextEdit()
        app_line_edit.setPlainText(app_text)
        self._set_text_edit_two_lines(app_line_edit)
        app_line_edit.setMinimumWidth(220)
        app_line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        app_line_edit.installEventFilter(self)
        default_text = self.app_name_A.toPlainText()
        if default_text:
            app_line_edit.setPlaceholderText(default_text)
        else:
            app_line_edit.setPlaceholderText(lang["app_name_placeholder"][lang_index])
        app_delete_btn = QPushButton(lang["delete"][lang_id])
        app_delete_btn.setMinimumHeight(36)
        app_delete_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._set_button_auto_width(app_delete_btn)
        app_delete_btn.clicked.connect(lambda checked, key=lang_key: self._remove_lang_row(key))
        app_row_layout.addWidget(app_label)
        app_row_layout.addSpacing(5)
        app_row_layout.addWidget(app_line_edit)
        app_row_layout.addWidget(app_delete_btn)
        app_row_layout.addStretch(1)
        self.app_name_extra_rows_layout.addLayout(app_row_layout)
        self.app_name_extra_rows[lang_key] = {
            "layout": app_row_layout,
            "label": app_label,
            "line_edit": app_line_edit,
            "delete_btn": app_delete_btn,
        }
        # 记录用户手动添加的语言
        if lang_key not in self.user_added_langs:
            self.user_added_langs.append(lang_key)
        # 内容变化后自动调整窗口高度，避免其他 Group 被拉伸
        QTimer.singleShot(50, self._adjust_height_to_content)

    def _remove_lang_row(self, lang_key):
        """删除应用名称的某一语言输入行。"""
        if lang_key not in self.extra_lang_keys:
            return
        self.extra_lang_keys.discard(lang_key)
        self.app_name_by_lang.pop(lang_key, None)
        if lang_key in self.user_added_langs:
            self.user_added_langs.remove(lang_key)

        def clear_row(widgets, rows_layout):
            layout = widgets["layout"]
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
            rows_layout.removeItem(layout)
            layout.deleteLater()

        if lang_key in self.app_name_extra_rows:
            clear_row(self.app_name_extra_rows.pop(lang_key), self.app_name_extra_rows_layout)

        self._refresh_lang_combos()
        # 内容变化后自动调整窗口高度，避免其他 Group 被拉伸
        QTimer.singleShot(50, self._adjust_height_to_content)

    def add_app_name_lang(self):
        """点击添加按钮，添加选中的语言行。"""
        index = self.app_name_lang_combo.currentIndex()
        if index < 0:
            return
        lang_key = self.app_name_lang_combo.itemData(index)
        # 从 TOML 配置中读取对应语言的文本，为空时才显示占位符（默认语言）
        app_text = self._conf["comm"].get(f"app_name_{lang_key}", "")
        self.app_name_by_lang.setdefault(lang_key, app_text)
        self._add_lang_row(lang_key, app_text)
        self._refresh_lang_combos()

    def save_conf(self):
        # 保存当前界面语言输入
        current_key = LANG_KEYS[self._current_app_name_lang_id]
        self.app_name_by_lang[current_key] = self.app_name_A.toPlainText()
        # 保存额外语言输入
        for key, widgets in self.app_name_extra_rows.items():
            self.app_name_by_lang[key] = widgets["line_edit"].toPlainText()
        # 写入 app_name_<key> 和 title_name_<key>
        for key in LANG_KEYS:
            name_value = self.app_name_by_lang.get(key, "")
            self._conf["comm"][f"app_name_{key}"] = name_value.replace("\n", "\\n")
            self._conf["comm"][f"title_name_{key}"] = name_value.replace("\n", " ").replace("\\n", "\\n")
        # 保存用户手动添加的语言列表
        self._conf["comm"]["user_added_langs"] = self.user_added_langs
        # 移除旧字段
        self._conf["comm"].pop("app_name_A", None)
        self._conf["comm"].pop("app_name_by_lang", None)
        self._conf["comm"].pop("title_A", None)

        with open(os.path.join(BASE_DIR, "app_conf.toml"), 'w', encoding='utf-8') as f:
            f.write(tomlkit.dumps(self._conf))

    def lang_changed(self, index):
        global lang_id
        selected_lang = self.lang_combo.itemText(index)
        print("当前语言:", selected_lang)
        lang_id = self.lang_combo.currentIndex()
        self.setWindowTitle(lang["app_title"][lang_id])
        self.export_btn.setText(lang["convert_and_package"][lang_id])
        if DEBUG:
            self.save_btn.setText(lang["save_config"][lang_id])
            self.pack_btn.setText(lang["pack_only"][lang_id])
        self.select_mode_group.setTitle(lang["select_mode"][lang_id])
        self.select_mode_label.setText(lang["input_source"][lang_id])
        self.user_dir_button.setText(lang["custom_directory"][lang_id])
        self.icon_button.setText(lang["select_icon"][lang_id])
        self.icon_group.setTitle(lang["icon"][lang_id])
        self.app_name_group.setTitle(lang["app_name"][lang_id])
        self.threshold_group.setTitle(lang["detection_threshold"][lang_id])
        self.add_app_name_lang_btn.setText(lang["add_other_language"][lang_id])
        self._set_button_auto_width(self.add_app_name_lang_btn)
        for widgets in self.app_name_extra_rows.values():
            widgets["delete_btn"].setText(lang["delete"][lang_id])
            self._set_button_auto_width(widgets["delete_btn"])
        # 切换默认语言：保存旧语言输入，加载新语言输入
        old_key = LANG_KEYS[self._current_app_name_lang_id]
        self.app_name_by_lang[old_key] = self.app_name_A.toPlainText()
        self._current_app_name_lang_id = lang_id
        new_key = LANG_KEYS[lang_id]
        self.app_name_A.setPlainText(self.app_name_by_lang.get(new_key, ""))
        self.app_name_current_lang_label.setText(LANG_DISPLAY_NAMES[lang_id])
        self._update_app_name_placeholder()

        # 重建额外语言行
        self._sync_from_config()
        self._refresh_lang_combos()

        # 切换语言时保留当前模式，避免 clear 触发 mode_changed 重置
        current_mode_index = self.mode_combo.currentIndex()
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.addItems(["MindPlus", lang["custom"][lang_id]])
        self.mode_combo.setCurrentIndex(current_mode_index if current_mode_index >= 0 else 0)
        self.mode_combo.blockSignals(False)
        self.mode_changed(self.mode_combo.currentIndex())

        self.zip_model_button.setText(lang["select_model_package"][lang_id])
        self.zip_dataset_button.setText(lang["select_dataset_package"][lang_id])


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
                QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], lang["model_zip_not_found"][lang_id])
                return
            extract_zip(model_zip, "model_input")
            #将data.yaml 改名为model.yaml
            os.rename(os.path.join("model_input", "data.yaml"), os.path.join("model_input", "model.yaml"))

            dataset_zip = self._conf["mindplus_options"]["dataset_zip"]
            if not dataset_zip or not os.path.exists(dataset_zip):
                print(f"数据集包不存在: {dataset_zip}")
                QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], lang["dataset_zip_not_found"][lang_id])
                return
            #extract_zip_without_top(dataset_zip, "model_input")
            extract_zip(dataset_zip, "model_input")
            self.model_dataset_dir = "model_input"
        else:
            #使用用户自定义目录
            self.model_dataset_dir = self._conf["user_options"]["user_dir"]
            if not self.model_dataset_dir or not os.path.exists(self.model_dataset_dir):
                print(f"用户自定义目录不存在: {self.model_dataset_dir}")
                QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], lang["user_dir_not_found"][lang_id])
                return
            model_yaml_path = os.path.join(self.model_dataset_dir, "model.yaml")
            if not os.path.exists(model_yaml_path):
                model_config = build_model_config_from_custom_dir(self.model_dataset_dir)
                write_model_yaml(model_yaml_path, model_config)
                print(f"已自动生成 model.yaml: {model_yaml_path}")

        onnx_path = os.path.join(self.model_dataset_dir, "best.onnx")
        if not os.path.exists(onnx_path):
            print(f"best.onnx 不存在: {onnx_path}")
            QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], lang["onnx_not_found"][lang_id])
            return

        if not self.app_name_A.toPlainText().strip():
            print(lang["app_name_cannot_be_empty"][lang_id])
            #弹出对话框
            QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], lang["app_name_cannot_be_empty"][lang_id])
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

        current_key = LANG_KEYS[self._current_app_name_lang_id]
        app_name_current_value = self.app_name_A.toPlainText().replace("\\n", "\n")
        application = clean_name(self.app_name_A.toPlainText())

        # 同步当前语言和额外语言输入到 app_name_by_lang
        self.app_name_by_lang[current_key] = app_name_current_value
        for key, widgets in self.app_name_extra_rows.items():
            self.app_name_by_lang[key] = widgets["line_edit"].toPlainText().replace("\\n", "\n")

        conf_data = copy.deepcopy(conf_template)
        conf_data["conf"]["application"] = application
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
        # 以系统语言为默认，缺失的语言用系统语言补齐
        system_key = self._get_system_lang_key()
        system_name = self.app_name_by_lang.get(system_key, app_name_current_value)
        desc_data["desc"]["application_name"] = {key: self.app_name_by_lang.get(key, system_name) for key in LANG_KEYS}
        # 标题使用对应的应用名称，并将换行替换为空格
        desc_data["desc"]["application_title"] = {key: self.app_name_by_lang.get(key, system_name).replace("\n", " ") for key in LANG_KEYS}

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

        # 用独立进程跑转换，避免 nncase 编译占用 GIL 导致 UI 卡死
        self._convert_output = ""
        kmodel_conf = os.path.join(BASE_DIR, "kmodel_conf.toml")
        if getattr(sys, "frozen", False):
            # PyInstaller 打包后 sys.executable 是程序自身，用 --run-convertor 进入转换子流程
            convert_args = ["--run-convertor", onnx_path, kmodel_path, self.dataset_path, kmodel_conf, output_zip]
        else:
            convertor_script = os.path.join(BASE_DIR, "convertor.py")
            convert_args = [convertor_script, onnx_path, kmodel_path, self.dataset_path, kmodel_conf, output_zip]
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_convert_output)
        self.process.finished.connect(self.on_conversion_finished)
        self.process.start(sys.executable, convert_args)
        print("正在转换")

    def _on_convert_output(self):
        # 把子进程的输出实时转发到控制台，同时缓存起来用于解析结果路径
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._convert_output += text
        print(text, end="", flush=True)

    def on_conversion_finished(self, exit_code, exit_status):
        self.export_btn.setText(lang["convert_and_package"][lang_id])
        file_path = None
        for line in self._convert_output.splitlines():
            if line.startswith("FINAL_ZIP="):
                file_path = line[len("FINAL_ZIP="):].strip()
        if exit_code == 0 and file_path:
            QMessageBox.information(
                self,
                lang["conversion_complete_title"][lang_id],
                lang["conversion_complete_message"][lang_id].format(path=file_path),
            )
            print(f"转换完成！文件路径: {file_path}")
        else:
            print(f"转换失败，退出码: {exit_code}")
            QMessageBox.warning(
                self,
                lang["dialog_warning_title"][lang_id],
                f"{lang['conversion_failed'][lang_id]} (exit code: {exit_code})",
            )

    def pack(self):
        # 打包 ZIP
        if not os.path.exists("model_output"):
            QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], lang["model_output_not_found"][lang_id])
            return

        required_files = ["conf.json", "desc.json"]
        missing = [f for f in required_files if not os.path.exists(os.path.join("model_output", f))]
        has_kmodel = any(f.endswith(".kmodel") for f in os.listdir("model_output"))
        if missing or not has_kmodel:
            print(f"输出目录缺少文件: {missing}, has_kmodel={has_kmodel}")
            QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], lang["model_output_not_found"][lang_id])
            return

        application = clean_name(self.app_name_A.toPlainText())
        try:
            final_zip_path = zip_with_md5(base_name=application)
            print("打包完成！")
            QMessageBox.information(
                self,
                lang["pack_complete_title"][lang_id],
                lang["pack_complete_message"][lang_id].format(path=final_zip_path),
            )
        except Exception as e:
            print(f"打包失败: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], f"{lang['pack_failed'][lang_id]}:\n{str(e)}")

# ============================================================================
# 程序启动方式说明
#
# 本程序有三种启动方式，由“是否 PyInstaller 打包”和“是否带 --run-convertor 参数”
# 两个条件组合决定：
#
# 1. python 运行 GUI 主程序（开发/调试时常用）：
#        python app.py
#    启动图形界面。点击“转换&打包”时，主进程会用 QProcess 启动子进程：
#        python convertor.py <onnx> <kmodel> <dataset> <toml> <output_zip>
#    转换在独立进程中进行（避免 nncase 编译占满 GIL 导致 UI 卡死），
#    子进程结束后打印 FINAL_ZIP=<路径>，主进程解析该路径并弹窗提示。
#
# 2. python 运行纯转换流程（不启动 GUI，等价于 GUI 点“转换&打包”的后半段）：
#        python convertor.py <onnx> <kmodel> <dataset> <toml> <output_zip>
#    直接走 convertor.main()，完成 转换 + 打包，不经过本文件的 __main__。
#
# 3. PyInstaller 打包成 exe 后（分发时用，打包命令：pyinstaller app.spec）：
#      - 主程序：  onnx2kmodel.exe
#        与方式 1 的 GUI 完全一致。区别仅在于点击“转换&打包”时，
#        由于 sys.executable 指向 exe 自身，子进程改为：
#            onnx2kmodel.exe --run-convertor <onnx> <kmodel> <dataset> <toml> <output_zip>
#      - 转换子进程：即下方 __main__ 里的 --run-convertor 分支，
#        检测到该参数后不创建 GUI，直接调用 convertor.main() 执行转换，
#        与方式 2 走的是同一个入口函数，保证两种模式下转换行为一致。
# ============================================================================
if __name__ == "__main__":
    # PyInstaller 打包后，转换子进程以 --run-convertor 参数重新启动本程序自身，
    # 此时直接进入转换流程，不创建 GUI（python 运行时则直接执行 convertor.py，不走这里）
    if "--run-convertor" in sys.argv:
        idx = sys.argv.index("--run-convertor")
        import convertor
        sys.exit(convertor.main(sys.argv[idx + 1:]))

    language_code = locale.getdefaultlocale()[0]
    print(f"默认语言环境: {language_code}")
    lang_id = 0
    if language_code:
        lc = language_code.lower()
        if lc.startswith("zh_cn") or lc.startswith("zh_sg"):
            lang_id = 1
        elif lc.startswith("zh_tw") or lc.startswith("zh_hk") or lc.startswith("zh_mo"):
            lang_id = 2
        elif lc.startswith("fr"):
            lang_id = 3
        elif lc.startswith("ko"):
            lang_id = 4
        elif lc.startswith("es"):
            lang_id = 5
        elif lc.startswith("pt"):
            lang_id = 6
        elif lc.startswith("ja"):
            lang_id = 7
    os.makedirs("model_output", exist_ok=True)
    os.makedirs("model_input", exist_ok=True)
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(palette=qdarkstyle.DarkPalette) + """
QGroupBox {
    margin-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding-top: 2px;
    padding-left: 4px;
    padding-right: 4px;
}
QPushButton {
    min-height: 28px;
}
""")
    window = ModelExportApp()
    window.setWindowOpacity(1.0)
    window.show()
    sys.exit(app.exec_())
