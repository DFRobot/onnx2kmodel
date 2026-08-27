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
import unicodedata
import time
from pathlib import Path
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit, QComboBox,
    QFileDialog, QSlider, QHBoxLayout, QVBoxLayout, QMessageBox, QGroupBox,
    QSizePolicy, QGridLayout, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QEvent, QTimer, QEventLoop
from PyQt5.QtGui import QIcon, QPalette, QPixmap, QColor
from PyQt5.QtCore import QProcess

from data_preparation import (
    PreparationError, SUPPORTED_SIZES, base_model_value,
    corresponding_label_directory, extract_mindplus_dataset,
    inspect_mindplus_source, prepare_yolo_source, reset_directory,
    sample_calibration_images, validate_training_source,
    write_model_yaml as write_prepared_model_yaml,
)

# 优先使用当前目录下的 qdarkstyle_dfrobot，避免依赖外部安装的包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#import qdarkstyle_dfrobot as qdarkstyle
import qdarkstyle

import io

# Debug 开关：True 时显示“保存配置”和“仅打包”按钮，False 时隐藏
DEBUG = False

# 打包后输出默认放在 exe 同目录，内置资源从 PyInstaller 临时资源目录读取。
IS_FROZEN = getattr(sys, "frozen", False)
BASE_DIR = os.path.dirname(sys.executable) if IS_FROZEN else os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = getattr(sys, "_MEIPASS", BASE_DIR)
TEMP_WORK_DIR = os.path.join(BASE_DIR, "HUSKYLENS 2 Package Generator Temp Files")

# PyInstaller 无控制台（windowed）模式下 sys.stdout 为 None，需要跳过
if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

lang_id = 0
lang = {
    "name": ["name", "名字", "名字", "nom", "이름", "nombre", "nome", "名前"],
    "language": ["Language", "语言", "語言", "Langue", "언어", "Idioma", "Idioma", "言語"],
    "data": ["Model ", "数据包", "數據包", "Modèle ", "모델 데이터 ", "Modelo ", "Pacote de modelo ", "モデル データ "],
    "icon": ["Icon", "图标", "圖標", "Icône", "아이콘", "Icono", "Ícone", "アイコン"],
    "preview": ["Preview", "预览", "預覽", "Aperçu", "미리보기", "Vista previa", "Visualização", "プレビュー"],
    "custom": ["YOLO", "YOLO", "YOLO", "YOLO", "YOLO", "YOLO", "YOLO", "YOLO"],
    "mindplus_dataset": ["MindPlus", "MindPlus", "MindPlus", "MindPlus", "MindPlus", "MindPlus", "MindPlus", "MindPlus"],
    "select_mode": ["Data Source", "数据源", "數據源", "Source de données", "데이터 소스", "Fuente de datos", "Fonte de dados", "データソース"],
    "input_source": ["Data Source", "数据来源", "資料來源", "Source des données", "데이터 소스", "Fuente de datos", "Fonte de dados", "データソース"],
    "output_directory": ["Output Directory", "输出目录", "輸出目錄", "Dossier de sortie", "출력 폴더", "Directorio de salida", "Diretório de saída", "出力フォルダー"],
    "select_output_directory": ["Select Installation Package Output Directory", "选择安装包输出目录", "選擇安裝包輸出目錄", "Sélectionner le dossier de sortie du paquet d’installation", "설치 패키지 출력 폴더 선택", "Seleccionar el directorio de salida del paquete de instalación", "Selecionar o diretório de saída do pacote de instalação", "インストールパッケージの出力フォルダーを選択"],
    "custom_directory": ["Dataset Folder", "数据集文件夹", "資料集資料夾", "Dossier du jeu de données", "데이터셋 폴더", "Carpeta del conjunto de datos", "Pasta do conjunto de dados", "データセットフォルダー"],
    "select_onnx_model": ["ONNX Model (.onnx)", "ONNX模型（.onnx）", "ONNX模型（.onnx）", "Modèle ONNX (.onnx)", "ONNX 모델 (.onnx)", "Modelo ONNX (.onnx)", "Modelo ONNX (.onnx)", "ONNXモデル（.onnx）"],
    "select_model_package": ["Model Package (.zip)", "模型包（.zip）", "模型包（.zip）", "Paquet modèle (.zip)", "모델 패키지 (.zip)", "Paquete de modelo (.zip)", "Pacote do modelo (.zip)", "モデルパッケージ（.zip）"],
    "select_dataset_package": ["Dataset Package (.zip)", "数据集包（.zip）", "資料集包（.zip）", "Paquet de données (.zip)", "데이터셋 패키지 (.zip)", "Paquete de datos (.zip)", "Pacote de dados (.zip)", "データパッケージ（.zip）"],
    "select_icon": ["Choose Icon", "选择图标", "選擇圖標", "Choisir l'icône", "아이콘 선택", "Elegir icono", "Escolher ícone", "アイコン選択"],
    "app_name": ["App Name", "应用名称", "應用名稱", "Nom de l'application", "앱 이름", "Nombre de la aplicación", "Nome do aplicativo", "アプリ名"],
    "add_other_language": ["Add other language", "添加其他语言", "添加其他語言", "Ajouter une langue", "다른 언어 추가", "Agregar otro idioma", "Adicionar outro idioma", "他の言語を追加"],
    "other_languages": ["Other languages", "其他语言", "其他語言", "Autres langues", "다른 언어", "Otros idiomas", "Outros idiomas", "他の言語"],
    "add": ["Add", "添加", "添加", "Ajouter", "추가", "Agregar", "Adicionar", "追加"],
    "delete": ["Delete", "删除", "刪除", "Supprimer", "삭제", "Eliminar", "Excluir", "削除"],
    "app_name_placeholder": [
        "Set the app name here. Press Enter to add a line break",
        "在此设置应用名称，按 Enter 键添加换行",
        "在此設定應用名稱，按 Enter 鍵新增換行",
        "Saisissez le nom ici, puis appuyez sur Entrée pour ajouter un retour",
        "여기에 앱 이름을 설정하고 Enter 키로 줄바꿈을 추가하세요",
        "Escribe aquí el nombre y pulsa Enter para añadir un salto de línea",
        "Defina o nome aqui e pressione Enter para adicionar uma quebra de linha",
        "ここでアプリ名を設定し、Enterキーで改行を追加します"
    ],
    "app_name_placeholder_short": [
        "Press Enter for a line break",
        "按 Enter 键添加换行",
        "按 Enter 鍵新增換行",
        "Appuyez sur Entrée pour un retour",
        "Enter로 줄바꿈 추가",
        "Pulsa Enter para un salto de línea",
        "Pressione Enter para quebra de linha",
        "Enterキーで改行を追加"
    ],
    "simplified_chinese": ["Simplified Chinese", "简体中文", "簡體中文", "Chinois simplifié", "간체 중국어", "Chino simplificado", "Chinês simplificado", "簡体中国語"],
    "traditional_chinese": ["Traditional Chinese", "繁体中文", "繁體中文", "Chinois traditionnel", "번체 중국어", "Chino tradicional", "Chinês tradicional", "繁体中国語"],
    "title_settings": ["Title Settings", "标题名称", "標題名稱", "Paramètres du titre", "제목 설정", "Configuración del título", "Configurações do título", "タイトル設定"],
    "detection_threshold": ["Default Threshold", "默认阈值", "預設閾值", "Seuil par défaut", "기본 임계값", "Umbral predeterminado", "Limite padrão", "デフォルトしきい値"],
    "save_config": ["Save Config", "保存配置", "保存配置", "Enregistrer config", "설정 저장", "Guardar configuración", "Salvar configuração", "設定を保存"],
    "convert_and_package": ["Convert and Package", "转换&打包", "轉換&打包", "Convertir et packager", "변환 및 패키징", "Convertir y empaquetar", "Converter e empacotar", "変換とパッケージ"],
    "start": ["Start", "开始", "開始", "Démarrer", "시작", "Iniciar", "Iniciar", "開始"],
    "pack_only": ["Pack Only", "仅打包", "僅打包", "Packager seul", "패키징만", "Solo empaquetar", "Apenas empacotar", "パッケージのみ"],
    "app_title": ["HUSKYLENS 2 Model Installation Package Generator", "HUSKYLENS 2 模型安装包生成器", "HUSKYLENS 2 模型安裝包產生器", "Générateur de paquets d’installation de modèles pour HUSKYLENS 2", "HUSKYLENS 2 모델 설치 패키지 생성기", "Generador de paquetes de instalación de modelos para HUSKYLENS 2", "Gerador de pacotes de instalação de modelos para HUSKYLENS 2", "HUSKYLENS 2 モデルインストールパッケージ生成ツール"],
    "select_custom_directory": ["Select Dataset Folder", "选择数据集文件夹", "選擇資料集資料夾", "Sélectionner le dossier du jeu de données", "데이터셋 폴더 선택", "Seleccionar carpeta del conjunto de datos", "Selecionar pasta do conjunto de dados", "データセットフォルダーを選択"],
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
    "model_series": ["Model Version", "模型版本", "模型版本", "Version du modèle", "모델 버전", "Versión del modelo", "Versão do modelo", "モデルバージョン"],
    "task_type": ["Task Type", "任务类型", "任務類型", "Type de tâche", "작업 유형", "Tipo de tarea", "Tipo de tarefa", "タスク種類"],
    "input_size": ["Input Size", "输入尺寸", "輸入尺寸", "Taille d’entrée", "입력 크기", "Tamaño de entrada", "Tamanho de entrada", "入力サイズ"],
    "unknown": ["Unknown", "未知", "未知", "Inconnu", "알 수 없음", "Desconocido", "Desconhecido", "不明"],
    "task_detect": ["Detection", "检测", "檢測", "Détection", "감지", "Detección", "Detecção", "検出"],
    "task_classify": ["Classification", "分类", "分類", "Classification", "분류", "Clasificación", "Classificação", "分類"],
    "task_segment": ["Segmentation", "分割", "分割", "Segmentation", "분할", "Segmentación", "Segmentação", "セグメンテーション"],
    "metadata_locked": ["This model information was read from ONNX metadata and cannot be changed.", "该模型信息已从 ONNX 元数据中获取，不可修改。", "該模型資訊已從 ONNX 元資料中取得，不可修改。", "Cette information provient des métadonnées ONNX et ne peut pas être modifiée.", "이 모델 정보는 ONNX 메타데이터에서 읽었으므로 변경할 수 없습니다.", "Esta información se obtuvo de los metadatos ONNX y no se puede modificar.", "Esta informação foi obtida dos metadados ONNX e não pode ser alterada.", "このモデル情報はONNXメタデータから取得されているため変更できません。"],
    "metadata_incomplete": ["Some model information could not be read from ONNX metadata. Select the unknown items manually.", "无法从 ONNX 模型中读取完整的模型信息，请手动选择尚未识别的项目。", "無法從 ONNX 模型中讀取完整的模型資訊，請手動選擇尚未識別的項目。", "Certaines informations ONNX sont introuvables. Sélectionnez manuellement les éléments inconnus.", "ONNX 메타데이터에서 일부 정보를 읽을 수 없습니다. 알 수 없는 항목을 직접 선택하세요.", "No se pudo leer toda la información ONNX. Seleccione manualmente los elementos desconocidos.", "Não foi possível ler todas as informações ONNX. Selecione manualmente os itens desconhecidos.", "ONNXメタデータから一部の情報を取得できません。未識別項目を手動で選択してください。"],
    "preparing_source": ["Checking and preparing the data source...", "正在检查并准备数据源…", "正在檢查並準備資料來源…", "Vérification et préparation de la source…", "데이터 소스를 확인하고 준비하는 중…", "Comprobando y preparando la fuente…", "Verificando e preparando a fonte…", "データソースを確認して準備しています…"],
    "source_ready": ["Preparation complete. Ready to start.", "准备工作已完成，可以开始。", "準備工作已完成，可以開始。", "Préparation terminée. Vous pouvez démarrer.", "준비가 완료되었습니다. 시작할 수 있습니다.", "Preparación terminada. Puede iniciar.", "Preparação concluída. Pode iniciar.", "準備が完了しました。開始できます。"],
    "app_name_required_to_start": ["Data source preparation is complete. Enter an app name to continue.", "数据源准备已完成，请输入应用名称后继续。", "資料來源準備已完成，請輸入應用名稱後繼續。", "La source est prête. Saisissez le nom de l’application pour continuer.", "데이터 소스 준비가 완료되었습니다. 계속하려면 앱 이름을 입력하세요.", "La fuente está preparada. Introduzca el nombre de la aplicación para continuar.", "A fonte está preparada. Digite o nome do aplicativo para continuar.", "データソースの準備が完了しました。続行するにはアプリ名を入力してください。"],
    "select_model_package_first": ["Select the model package first.", "请先选择模型包。", "請先選擇模型包。", "Sélectionnez d’abord le paquet modèle.", "먼저 모델 패키지를 선택하세요.", "Seleccione primero el paquete del modelo.", "Selecione primeiro o pacote do modelo.", "先にモデルパッケージを選択してください。"],
    "select_dataset_package_first": ["Select the dataset package next.", "接下来请选择数据集包。", "接下來請選擇資料集包。", "Sélectionnez ensuite le paquet de données.", "다음으로 데이터셋 패키지를 선택하세요.", "A continuación, seleccione el paquete de datos.", "Em seguida, selecione o pacote de dados.", "次にデータセットパッケージを選択してください。"],
    "select_source_directory_first": ["Select the file directory first.", "请先选择文件目录。", "請先選擇檔案目錄。", "Sélectionnez d’abord le répertoire.", "먼저 파일 폴더를 선택하세요.", "Seleccione primero el directorio.", "Selecione primeiro o diretório.", "先にフォルダーを選択してください。"],
    "select_dataset_folder_first": ["Select the dataset folder first.", "请先选择数据集文件夹。", "請先選擇資料集資料夾。", "Sélectionnez d’abord le dossier du jeu de données.", "먼저 데이터셋 폴더를 선택하세요.", "Seleccione primero la carpeta del conjunto de datos.", "Selecione primeiro a pasta do conjunto de dados.", "先にデータセットフォルダーを選択してください。"],
    "select_onnx_model_first": ["Next, select the ONNX model.", "接下来请选择 ONNX 模型。", "接下來請選擇 ONNX 模型。", "Sélectionnez ensuite le modèle ONNX.", "다음으로 ONNX 모델을 선택하세요.", "A continuación, seleccione el modelo ONNX.", "Em seguida, selecione o modelo ONNX.", "次にONNXモデルを選択してください。"],
    "sampling_images": ["Selecting calibration images...", "正在抽取校准图片…", "正在抽取校準圖片…", "Sélection des images d’étalonnage…", "보정 이미지를 선택하는 중…", "Seleccionando imágenes de calibración…", "Selecionando imagens de calibração…", "キャリブレーション画像を抽出しています…"],
    "preprocessing_images": ["Preprocessing calibration images...", "正在预处理校准图片…", "正在預處理校準圖片…", "Prétraitement des images d’étalonnage…", "보정 이미지를 전처리하는 중…", "Preprocesando imágenes de calibración…", "Pré-processando imagens de calibração…", "キャリブレーション画像を前処理しています…"],
    "importing_onnx": ["Importing the ONNX model...", "正在导入 ONNX 模型…", "正在匯入 ONNX 模型…", "Importation du modèle ONNX…", "ONNX 모델을 가져오는 중…", "Importando el modelo ONNX…", "Importando o modelo ONNX…", "ONNXモデルをインポートしています…"],
    "preparing_quantization": ["Preparing model quantization...", "正在准备模型量化…", "正在準備模型量化…", "Préparation de la quantification…", "모델 양자화를 준비하는 중…", "Preparando la cuantización del modelo…", "Preparando a quantização do modelo…", "モデル量子化を準備しています…"],
    "quantizing_compiling": ["Quantizing and compiling the model. This may take a while...", "正在量化并编译模型，此阶段耗时较长…", "正在量化並編譯模型，此階段耗時較長…", "Quantification et compilation du modèle, cette étape peut prendre du temps…", "모델을 양자화하고 컴파일하는 중입니다. 시간이 걸릴 수 있습니다…", "Cuantizando y compilando el modelo; esta etapa puede tardar…", "Quantizando e compilando o modelo; esta etapa pode demorar…", "モデルを量子化してコンパイルしています。この処理には時間がかかります…"],
    "generating_kmodel": ["Generating the KModel...", "正在生成 KModel…", "正在產生 KModel…", "Génération du KModel…", "KModel을 생성하는 중…", "Generando el KModel…", "Gerando o KModel…", "KModelを生成しています…"],
    "converting_model": ["Converting model...", "正在转换模型…", "正在轉換模型…", "Conversion du modèle…", "모델 변환 중…", "Convirtiendo el modelo…", "Convertendo o modelo…", "モデルを変換しています…"],
    "generating_package": ["Generating installation package...", "正在生成安装包…", "正在產生安裝包…", "Génération du paquet d’installation…", "설치 패키지 생성 중…", "Generando el paquete de instalación…", "Gerando o pacote de instalação…", "インストールパッケージを生成しています…"],
    "package_generated": ["Installation package generated.", "安装包生成完成。", "安裝包產生完成。", "Paquet d’installation généré.", "설치 패키지가 생성되었습니다.", "Paquete de instalación generado.", "Pacote de instalação gerado.", "インストールパッケージが生成されました。"],
    "clear_cache_question": ["The installation package was generated. Clear the cache files from this run?", "安装包已生成。是否清除本次任务产生的缓存文件？", "安裝包已產生。是否清除本次任務產生的快取檔案？", "Le paquet est généré. Effacer le cache de cette exécution ?", "설치 패키지가 생성되었습니다. 이번 작업의 캐시 파일을 삭제할까요?", "El paquete se generó. ¿Borrar los archivos de caché de esta ejecución?", "O pacote foi gerado. Limpar os arquivos de cache desta execução?", "パッケージが生成されました。今回のキャッシュを削除しますか？"],
    "open_files": ["Open Files", "打开文件", "開啟檔案", "Ouvrir les fichiers", "파일 열기", "Abrir archivos", "Abrir arquivos", "ファイルを開く"],
    "cancel": ["Cancel", "取消", "取消", "Annuler", "취소", "Cancelar", "Cancelar", "キャンセル"],
    "clear": ["Clear", "清除", "清除", "Effacer", "삭제", "Borrar", "Limpar", "消去"],
    "cache_cleared": ["Cache files cleared.", "缓存文件已清除。", "快取檔案已清除。", "Fichiers cache effacés.", "캐시 파일이 삭제되었습니다.", "Archivos de caché borrados.", "Arquivos de cache limpos.", "キャッシュを消去しました。"],
    "preparation_failed": ["Data source preparation failed", "数据源准备失败", "資料來源準備失敗", "Échec de préparation de la source", "데이터 소스 준비 실패", "Error al preparar la fuente", "Falha ao preparar a fonte", "データソースの準備に失敗しました"],
    "yaml_missing_message": ["No dataset configuration was found. Put one correct .yaml dataset configuration file in the selected folder.", "未找到数据集配置文件，请在所选文件夹中放置一个正确的 .yaml 数据集配置文件。", "未找到資料集設定檔，請在所選資料夾中放置一個正確的 .yaml 資料集設定檔。", "Aucune configuration de jeu de données trouvée. Placez un seul fichier .yaml correct dans le dossier sélectionné.", "데이터셋 설정 파일을 찾을 수 없습니다. 선택한 폴더에 올바른 .yaml 파일 하나를 넣으세요.", "No se encontró la configuración del conjunto de datos. Coloque un archivo .yaml correcto en la carpeta seleccionada.", "A configuração do conjunto de dados não foi encontrada. Coloque um arquivo .yaml correto na pasta selecionada.", "データセット設定が見つかりません。選択したフォルダーに正しい.yamlファイルを1つ配置してください。"],
    "yaml_multiple_message": ["Multiple .yaml files were found. Keep only one correct dataset configuration file and try again.", "检测到多个 .yaml 文件，无法确定数据集配置。请仅保留一个正确的数据集配置文件后重试。", "偵測到多個 .yaml 檔案，無法確定資料集設定。請只保留一個正確的設定檔後重試。", "Plusieurs fichiers .yaml ont été trouvés. Conservez un seul fichier de configuration correct.", "여러 .yaml 파일이 발견되었습니다. 올바른 데이터셋 설정 파일 하나만 남기고 다시 시도하세요.", "Se encontraron varios archivos .yaml. Conserve solo un archivo de configuración correcto.", "Foram encontrados vários arquivos .yaml. Mantenha apenas um arquivo de configuração correto.", "複数の.yamlファイルが見つかりました。正しい設定ファイルを1つだけ残してください。"],
    "yaml_fields_message": ["The dataset configuration is missing required fields: {fields}.", "数据集配置文件缺少以下必要信息：{fields}。", "資料集設定檔缺少以下必要資訊：{fields}。", "La configuration du jeu de données ne contient pas les champs requis : {fields}.", "데이터셋 설정에 필수 항목이 없습니다: {fields}.", "Faltan campos obligatorios en la configuración: {fields}.", "Faltam campos obrigatórios na configuração: {fields}.", "データセット設定に必須項目がありません：{fields}。"],
    "onnx_missing_message": ["No ONNX model was found. Put one correct .onnx model in the selected folder.", "未找到 ONNX 模型，请在所选文件夹中放入一个正确的 .onnx 文件。", "未找到 ONNX 模型，請在所選資料夾中放入一個正確的 .onnx 檔案。", "Aucun modèle ONNX trouvé. Placez un fichier .onnx correct dans le dossier sélectionné.", "ONNX 모델을 찾을 수 없습니다. 선택한 폴더에 올바른 .onnx 파일 하나를 넣으세요.", "No se encontró un modelo ONNX. Coloque un archivo .onnx correcto en la carpeta seleccionada.", "Nenhum modelo ONNX foi encontrado. Coloque um arquivo .onnx correto na pasta selecionada.", "ONNXモデルが見つかりません。選択したフォルダーに正しい.onnxファイルを1つ配置してください。"],
    "onnx_multiple_message": ["Multiple .onnx files were found. Keep only one correct ONNX model and try again.", "检测到多个 .onnx 文件，无法确定需要转换的模型。请仅保留一个正确的 ONNX 模型后重试。", "偵測到多個 .onnx 檔案，無法確定要轉換的模型。請只保留一個正確的 ONNX 模型後重試。", "Plusieurs fichiers .onnx ont été trouvés. Conservez un seul modèle ONNX correct.", "여러 .onnx 파일이 발견되었습니다. 올바른 ONNX 모델 하나만 남기고 다시 시도하세요.", "Se encontraron varios archivos .onnx. Conserve solo un modelo ONNX correcto.", "Foram encontrados vários arquivos .onnx. Mantenha apenas um modelo ONNX correto.", "複数の.onnxファイルが見つかりました。正しいONNXモデルを1つだけ残してください。"],
    "unsupported_model_message": ["Only YOLOv8n and YOLO11n are supported. Unsupported model: {model}", "当前仅支持 YOLOv8n 和 YOLO11n，该 ONNX 模型属于不受支持的模型系列：{model}", "目前只支援 YOLOv8n 和 YOLO11n，該 ONNX 模型屬於不支援的模型系列：{model}", "Seuls YOLOv8n et YOLO11n sont pris en charge. Modèle non pris en charge : {model}", "YOLOv8n과 YOLO11n만 지원됩니다. 지원되지 않는 모델: {model}", "Solo se admiten YOLOv8n y YOLO11n. Modelo no compatible: {model}", "Somente YOLOv8n e YOLO11n são suportados. Modelo não suportado: {model}", "YOLOv8nとYOLO11nのみ対応しています。未対応モデル：{model}"],
    "unsupported_size_message": ["Only 224 × 224, 320 × 320 and 640 × 640 are supported. Unsupported size: {size}", "当前仅支持 224 × 224、320 × 320 和 640 × 640，该模型输入尺寸不受支持：{size}", "目前只支援 224 × 224、320 × 320 和 640 × 640，該模型輸入尺寸不受支援：{size}", "Seules les tailles 224 × 224, 320 × 320 et 640 × 640 sont prises en charge. Taille non prise en charge : {size}", "224 × 224, 320 × 320, 640 × 640만 지원됩니다. 지원되지 않는 크기: {size}", "Solo se admiten 224 × 224, 320 × 320 y 640 × 640. Tamaño no compatible: {size}", "Somente 224 × 224, 320 × 320 e 640 × 640 são suportados. Tamanho não suportado: {size}", "224 × 224、320 × 320、640 × 640のみ対応しています。未対応サイズ：{size}"],
    "train_missing_message": ["Training image directory not found:\n{path}", "未找到训练图片目录：\n{path}\n\n请确认 data.yaml 中的 path 和 train 设置正确。", "未找到訓練圖片目錄：\n{path}\n\n請確認 data.yaml 中的 path 和 train 設定正確。", "Dossier d’images d’entraînement introuvable :\n{path}", "학습 이미지 폴더를 찾을 수 없습니다:\n{path}", "No se encontró el directorio de imágenes de entrenamiento:\n{path}", "Diretório de imagens de treinamento não encontrado:\n{path}", "学習画像フォルダーが見つかりません：\n{path}"],
    "label_missing_message": ["Detection/segmentation label directory not found:\n{path}", "未找到检测/分割训练标签目录：\n{path}\n\n请确认 data.yaml 中的 path 和 train 设置正确，并保证训练图片目录与训练标签目录对应。", "未找到檢測/分割訓練標籤目錄：\n{path}\n\n請確認 data.yaml 設定正確，且圖片與標籤目錄相互對應。", "Dossier d’étiquettes de détection/segmentation introuvable :\n{path}", "감지/분할 학습 라벨 폴더를 찾을 수 없습니다:\n{path}", "No se encontró el directorio de etiquetas de detección/segmentación:\n{path}", "Diretório de rótulos de detecção/segmentação não encontrado:\n{path}", "検出・セグメンテーション用ラベルフォルダーが見つかりません：\n{path}"],
    "classes_uncovered_message": ["No calibration images were found for these classes: {classes}", "以下类别没有找到可用校准图片：{classes}", "以下類別找不到可用的校準圖片：{classes}", "Aucune image d’étalonnage trouvée pour ces classes : {classes}", "다음 클래스의 보정 이미지를 찾을 수 없습니다: {classes}", "No se encontraron imágenes de calibración para estas clases: {classes}", "Não foram encontradas imagens de calibração para estas classes: {classes}", "次のクラスのキャリブレーション画像が見つかりません：{classes}"],
    "yaml_invalid_message": ["The dataset configuration is invalid:\n{path}\n\n{reason}", "数据集配置文件无效：\n{path}\n\n{reason}", "資料集設定檔無效：\n{path}\n\n{reason}", "La configuration du jeu de données est invalide :\n{path}\n\n{reason}", "데이터셋 설정 파일이 잘못되었습니다:\n{path}\n\n{reason}", "La configuración del conjunto de datos no es válida:\n{path}\n\n{reason}", "A configuração do conjunto de dados é inválida:\n{path}\n\n{reason}", "データセット設定が無効です：\n{path}\n\n{reason}"],
    "yaml_path_invalid_message": ["The path and train fields must contain valid path text.", "data.yaml 中的 path 和 train 必须是有效路径文本。", "data.yaml 中的 path 和 train 必須是有效路徑文字。", "Les champs path et train doivent contenir des chemins valides.", "path와 train 항목은 올바른 경로 문자열이어야 합니다.", "Los campos path y train deben contener rutas válidas.", "Os campos path e train devem conter caminhos válidos.", "pathとtrainには有効なパス文字列が必要です。"],
    "names_invalid_message": ["The names field is empty or invalid.", "data.yaml 中的 names 信息为空或格式不正确。", "data.yaml 中的 names 資訊為空或格式不正確。", "Le champ names est vide ou invalide.", "names 항목이 비어 있거나 형식이 잘못되었습니다.", "El campo names está vacío o no es válido.", "O campo names está vazio ou é inválido.", "namesが空か、形式が正しくありません。"],
    "onnx_invalid_message": ["The ONNX model cannot be read:\n{path}\n\n{reason}", "无法读取 ONNX 模型：\n{path}\n\n{reason}", "無法讀取 ONNX 模型：\n{path}\n\n{reason}", "Impossible de lire le modèle ONNX :\n{path}\n\n{reason}", "ONNX 모델을 읽을 수 없습니다:\n{path}\n\n{reason}", "No se puede leer el modelo ONNX:\n{path}\n\n{reason}", "Não foi possível ler o modelo ONNX:\n{path}\n\n{reason}", "ONNXモデルを読み込めません：\n{path}\n\n{reason}"],
    "source_missing_message": ["The selected data source does not exist:\n{path}", "所选数据源不存在：\n{path}", "所選資料來源不存在：\n{path}", "La source de données sélectionnée n’existe pas :\n{path}", "선택한 데이터 소스가 없습니다:\n{path}", "La fuente seleccionada no existe:\n{path}", "A fonte selecionada não existe:\n{path}", "選択したデータソースが存在しません：\n{path}"],
    "train_images_empty_message": ["No usable images were found in:\n{path}", "训练图片目录中没有可用图片：\n{path}", "訓練圖片目錄中沒有可用圖片：\n{path}", "Aucune image utilisable dans :\n{path}", "사용 가능한 학습 이미지가 없습니다:\n{path}", "No hay imágenes utilizables en:\n{path}", "Não há imagens utilizáveis em:\n{path}", "使用可能な学習画像がありません：\n{path}"],
    "label_path_message": ["The label directory cannot be derived from:\n{train_path}\n\nThe training path must contain an images folder.", "无法根据训练图片路径定位标签目录：\n{train_path}\n\n请确保路径中包含 images 目录。", "無法根據訓練圖片路徑定位標籤目錄：\n{train_path}\n\n請確認路徑中包含 images 目錄。", "Impossible de déterminer le dossier labels depuis :\n{train_path}\n\nLe chemin doit contenir un dossier images.", "다음 경로에서 라벨 폴더를 찾을 수 없습니다:\n{train_path}\n\n경로에 images 폴더가 있어야 합니다.", "No se puede localizar labels desde:\n{train_path}\n\nLa ruta debe contener una carpeta images.", "Não foi possível localizar labels a partir de:\n{train_path}\n\nO caminho deve conter a pasta images.", "次のパスからlabelsフォルダーを特定できません：\n{train_path}\n\nパスにimagesフォルダーが必要です。"],
    "label_files_empty_message": ["No .txt label files were found in:\n{path}", "训练标签目录中没有 .txt 标签文件：\n{path}", "訓練標籤目錄中沒有 .txt 標籤檔案：\n{path}", "Aucun fichier d’étiquette .txt dans :\n{path}", ".txt 라벨 파일이 없습니다:\n{path}", "No hay archivos de etiquetas .txt en:\n{path}", "Não há arquivos de rótulo .txt em:\n{path}", ".txtラベルファイルがありません：\n{path}"],
    "pair_missing_message": ["No matching image and label names were found.\nImages: {image_path}\nLabels: {label_path}", "没有找到同名的训练图片与标签。\n图片目录：{image_path}\n标签目录：{label_path}", "找不到同名的訓練圖片與標籤。\n圖片目錄：{image_path}\n標籤目錄：{label_path}", "Aucune paire image/étiquette de même nom.\nImages : {image_path}\nÉtiquettes : {label_path}", "이름이 같은 이미지와 라벨을 찾을 수 없습니다.\n이미지: {image_path}\n라벨: {label_path}", "No hay imágenes y etiquetas con el mismo nombre.\nImágenes: {image_path}\nEtiquetas: {label_path}", "Não há imagens e rótulos com o mesmo nome.\nImagens: {image_path}\nRótulos: {label_path}", "同名の画像とラベルが見つかりません。\n画像：{image_path}\nラベル：{label_path}"],
    "class_dir_missing_message": ["No classification folder was found for “{class_name}” in:\n{path}", "未找到类别“{class_name}”对应的分类目录：\n{path}", "找不到類別「{class_name}」對應的分類目錄：\n{path}", "Dossier de classe introuvable pour « {class_name} » dans :\n{path}", "“{class_name}” 클래스 폴더를 찾을 수 없습니다:\n{path}", "No se encontró la carpeta de la clase “{class_name}” en:\n{path}", "A pasta da classe “{class_name}” não foi encontrada em:\n{path}", "クラス「{class_name}」のフォルダーがありません：\n{path}"],
    "class_images_empty_message": ["No usable images were found for “{class_name}” in:\n{path}", "类别“{class_name}”目录中没有可用图片：\n{path}", "類別「{class_name}」目錄中沒有可用圖片：\n{path}", "Aucune image utilisable pour « {class_name} » dans :\n{path}", "“{class_name}” 클래스에 사용 가능한 이미지가 없습니다:\n{path}", "No hay imágenes utilizables para “{class_name}” en:\n{path}", "Não há imagens utilizáveis para “{class_name}” em:\n{path}", "クラス「{class_name}」に使用可能な画像がありません：\n{path}"],
    "mindplus_yaml_missing_message": ["The MindPlus model package has no model .yaml file.", "MindPlus 模型包中缺少模型配置 .yaml 文件。", "MindPlus 模型包中缺少模型設定 .yaml 檔案。", "Le paquet modèle MindPlus ne contient pas de fichier .yaml.", "MindPlus 모델 패키지에 .yaml 설정 파일이 없습니다.", "Falta el archivo .yaml en el paquete de modelo MindPlus.", "O pacote de modelo MindPlus não contém o arquivo .yaml.", "MindPlusモデルパッケージに.yaml設定がありません。"],
    "mindplus_yaml_multiple_message": ["The MindPlus model package has multiple .yaml files. Keep one model configuration.", "MindPlus 模型包中存在多个 .yaml 文件，无法确定模型配置。", "MindPlus 模型包中存在多個 .yaml 檔案，無法確定模型設定。", "Le paquet modèle MindPlus contient plusieurs fichiers .yaml.", "MindPlus 모델 패키지에 여러 .yaml 파일이 있습니다.", "El paquete de modelo MindPlus contiene varios archivos .yaml.", "O pacote de modelo MindPlus contém vários arquivos .yaml.", "MindPlusモデルパッケージに複数の.yamlファイルがあります。"],
    "class_mismatch_message": ["The class information in the MindPlus model and dataset packages does not match.", "MindPlus 模型包与数据集包中的类别信息不一致。", "MindPlus 模型包與資料集包中的類別資訊不一致。", "Les classes des paquets modèle et données MindPlus ne correspondent pas.", "MindPlus 모델과 데이터셋 패키지의 클래스 정보가 일치하지 않습니다.", "Las clases de los paquetes MindPlus no coinciden.", "As classes dos pacotes MindPlus não correspondem.", "MindPlusのモデルとデータセットでクラス情報が一致しません。"],
    "classification_class_count_mismatch_message": ["The classification model has {model_count} output classes, but the dataset defines {dataset_count} classes. Make the class counts consistent and try again.", "分类模型包含 {model_count} 个输出类别，但数据集中定义了 {dataset_count} 个类别。请确保类别数量一致后重试。", "分類模型包含 {model_count} 個輸出類別，但資料集中定義了 {dataset_count} 個類別。請確保類別數量一致後重試。", "Le modèle de classification produit {model_count} classes, mais le jeu de données en définit {dataset_count}. Corrigez le nombre de classes puis réessayez.", "분류 모델의 출력 클래스는 {model_count}개이지만 데이터셋에는 {dataset_count}개 클래스가 정의되어 있습니다. 클래스 수를 일치시킨 후 다시 시도하세요.", "El modelo de clasificación tiene {model_count} clases de salida, pero el conjunto de datos define {dataset_count}. Haga coincidir las cantidades y vuelva a intentarlo.", "O modelo de classificação tem {model_count} classes de saída, mas o conjunto de dados define {dataset_count}. Ajuste as quantidades e tente novamente.", "分類モデルの出力クラスは {model_count} 個ですが、データセットには {dataset_count} 個のクラスが定義されています。数を一致させて再試行してください。"],
    "classification_class_count_conflict_message": ["The class count in the ONNX metadata conflicts with the model output shape. Export the model again or check the ONNX file.", "ONNX 元数据中的类别数量与模型输出维度不一致，请重新导出模型或检查 ONNX 文件。", "ONNX 元資料中的類別數量與模型輸出維度不一致，請重新匯出模型或檢查 ONNX 檔案。", "Le nombre de classes des métadonnées ONNX ne correspond pas à la sortie du modèle. Réexportez ou vérifiez le fichier ONNX.", "ONNX 메타데이터의 클래스 수와 모델 출력 크기가 일치하지 않습니다. 모델을 다시 내보내거나 ONNX 파일을 확인하세요.", "La cantidad de clases de los metadatos ONNX no coincide con la salida del modelo. Vuelva a exportar o revise el archivo ONNX.", "A quantidade de classes nos metadados ONNX não corresponde à saída do modelo. Exporte novamente ou verifique o arquivo ONNX.", "ONNXメタデータのクラス数とモデル出力形状が一致しません。モデルを再エクスポートするかONNXファイルを確認してください。"],
    "classification_class_count_unavailable_message": ["The number of classes could not be confirmed from the ONNX model. The dataset class definitions will be used; verify that their count and order match model training.", "无法从 ONNX 模型确认类别数量，将使用数据集中的类别定义。请确认类别数量和顺序与模型训练时一致。", "無法從 ONNX 模型確認類別數量，將使用資料集中的類別定義。請確認類別數量和順序與模型訓練時一致。", "Le nombre de classes n’a pas pu être confirmé depuis le modèle ONNX. Les classes du jeu de données seront utilisées ; vérifiez leur nombre et leur ordre.", "ONNX 모델에서 클래스 수를 확인할 수 없어 데이터셋의 클래스 정의를 사용합니다. 클래스 수와 순서가 학습 시와 같은지 확인하세요.", "No se pudo confirmar la cantidad de clases desde el modelo ONNX. Se usarán las clases del conjunto de datos; compruebe su cantidad y orden.", "Não foi possível confirmar a quantidade de classes no modelo ONNX. As classes do conjunto de dados serão usadas; verifique a quantidade e a ordem.", "ONNXモデルからクラス数を確認できないため、データセットのクラス定義を使用します。学習時とクラス数・順序が一致することを確認してください。"],
    "unsafe_zip_message": ["The ZIP contains an unsafe path: {file}", "ZIP 中包含不安全的文件路径：{file}", "ZIP 中包含不安全的檔案路徑：{file}", "Le ZIP contient un chemin non sécurisé : {file}", "ZIP에 안전하지 않은 경로가 있습니다: {file}", "El ZIP contiene una ruta no segura: {file}", "O ZIP contém um caminho inseguro: {file}", "ZIPに安全でないパスが含まれています：{file}"],
    "staging_cleanup_failed": ["Unable to clear the MindPlus staging cache:\n{path}\n\n{reason}", "无法清理 MindPlus 隔离缓存目录：\n{path}\n\n{reason}", "無法清除 MindPlus 隔離快取目錄：\n{path}\n\n{reason}", "Impossible d’effacer le cache temporaire MindPlus :\n{path}\n\n{reason}", "MindPlus 임시 캐시를 삭제할 수 없습니다:\n{path}\n\n{reason}", "No se puede borrar la caché temporal de MindPlus:\n{path}\n\n{reason}", "Não foi possível limpar o cache temporário do MindPlus:\n{path}\n\n{reason}", "MindPlus一時キャッシュを消去できません：\n{path}\n\n{reason}"],
    "operation_failed_message": ["Operation failed:\n{reason}", "操作失败：\n{reason}", "操作失敗：\n{reason}", "Échec de l’opération :\n{reason}", "작업 실패:\n{reason}", "La operación falló:\n{reason}", "Falha na operação:\n{reason}", "操作に失敗しました：\n{reason}"],
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


def package_base_name(application, source_info):
    """生成“应用名-模型版本-任务类型-输入尺寸”的安装包基础文件名。"""
    series = source_info["series"]
    task = source_info["task"]
    size = source_info["size"]
    input_size = size[0] if isinstance(size, (list, tuple)) else size
    task_short = {"detect": "det", "classify": "cls", "segment": "seg"}[task]
    return f"{application}-{series}-{task_short}-{input_size}"

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
        self._conversion_running = False
        self._status_flash_step = 0
        self._status_flash_timer = QTimer(self)
        self._status_flash_timer.setInterval(140)
        self._status_flash_timer.timeout.connect(self._update_status_flash)
        self._app_event_filter_installed = False
        self._source_info = None
        self._source_ready = False
        self._model_info_locked = {"series": False, "task": False, "size": False}
        self._model_info_updating = False
        self._current_stage = "idle"
        self._stage_started_at = 0.0
        self._active_progress_step = 0
        self._completed_progress_steps = set()
        self._last_preparation_error = None
        self.setWindowTitle(lang["app_title"][lang_id])
        self.resize(620, 560)
        self.setMinimumWidth(580)
        with open(os.path.join(RESOURCE_DIR, "app_conf.toml"), 'r', encoding='utf-8') as f:
            self._conf = tomlkit.parse(f.read())
            print(self._conf)
        self._reset_startup_state()
        self._ensure_default_icon()
        self.output_dir = BASE_DIR
        self.work_dir = os.path.join(self.output_dir, "HUSKYLENS 2 Package Generator Temp Files")
        self._normalize_app_name_newlines()
        self._ensure_app_name_keys()
        self.init_ui()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_event_filter_installed = True

    def _reset_startup_state(self):
        """每次启动都恢复默认输入，不沿用上一次运行中的选择。"""
        comm = self._conf["comm"]
        comm["mode"] = "User"
        comm["det_threshold"] = 0.6
        comm["user_added_langs"] = []
        for key in LANG_KEYS:
            comm[f"app_name_{key}"] = ""
            comm[f"title_name_{key}"] = ""
        comm.pop("app_name_A", None)
        comm.pop("app_name_by_lang", None)
        comm.pop("title_A", None)

        self._conf["mindplus_options"]["model_zip"] = ""
        self._conf["mindplus_options"]["dataset_zip"] = ""
        self._conf["user_options"]["user_dir"] = ""
        self._conf["user_options"]["onnx_file"] = ""

    def _ensure_default_icon(self):
        """预览使用 icon.png，窗口和任务栏使用与 EXE 一致的 exe_icon.png。"""
        default_icon = os.path.join(RESOURCE_DIR, "icon.png")
        self._conf["comm"]["icon_file"] = default_icon if os.path.exists(default_icon) else ""
        app_icon = os.path.join(RESOURCE_DIR, "exe_icon.png")
        if not os.path.exists(app_icon):
            app_icon = default_icon
        if os.path.exists(app_icon):
            window_icon = QIcon(app_icon)
            self.setWindowIcon(window_icon)
            app = QApplication.instance()
            if app is not None:
                app.setWindowIcon(window_icon)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._apply_windows_title_bar_style)

    def _apply_windows_title_bar_style(self):
        """让 Windows 原生标题栏使用与应用界面一致的深色配色。"""
        if sys.platform != "win32":
            return
        try:
            import ctypes

            hwnd = int(self.winId())
            dwmapi = ctypes.windll.dwmapi

            dark_mode = ctypes.c_int(1)
            # Windows 11 使用 20，部分 Windows 10 版本使用 19。
            for attribute in (20, 19):
                result = dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    attribute,
                    ctypes.byref(dark_mode),
                    ctypes.sizeof(dark_mode),
                )
                if result == 0:
                    break

            def colorref(color):
                return ctypes.c_uint(color.red() | (color.green() << 8) | (color.blue() << 16))

            palette = self.palette()
            caption_color = colorref(palette.color(QPalette.Window))
            text_color = colorref(palette.color(QPalette.WindowText))
            border_color = colorref(palette.color(QPalette.Mid))
            for attribute, value in (
                (35, caption_color),
                (36, text_color),
                (34, border_color),
            ):
                dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    attribute,
                    ctypes.byref(value),
                    ctypes.sizeof(value),
                )
        except (AttributeError, OSError):
            # 较旧 Windows 不支持这些属性时继续使用系统标题栏。
            pass

    def _normalize_app_name_newlines(self):
        """将配置中应用名称的 \\n 转义序列转换为内部使用的真实换行符。"""
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
        app = QApplication.instance()
        if app is not None and self._app_event_filter_installed:
            app.removeEventFilter(self)
            self._app_event_filter_installed = False
        # 停止所有未完成的 QTimer
        for child in self.findChildren(QTimer):
            if child.isActive():
                child.stop()
        # 确保事件循环退出
        if app is not None:
            app.quit()
        super().closeEvent(event)

    def _adjust_height_to_content(self):
        """根据当前布局高度调整窗口高度，保持宽度不变。"""
        self.layout().update()
        self.resize(self.width(), self.sizeHint().height())

    def init_ui(self):
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(16, 14, 16, 14)
        self.main_layout.setSpacing(10)

        # --- 1. 顶部界面语言 ---
        self.language_label = QLabel(lang["language"][lang_id])
        self.language_label.setFixedWidth(150)
        self.language_label.setStyleSheet("font-weight: bold;")
        self.lang_combo = QComboBox()
        self.lang_combo.setFixedHeight(36)
        self.lang_combo.addItems(["English", "简体中文", "繁体中文", "Français", "한국어", "Español", "Português (Brasil)", "日本語"])
        self.lang_combo.view().setSpacing(2)
        self.lang_combo.view().setStyleSheet("QListView::item { min-height: 36px; padding: 2px 8px; }")
        self.lang_combo.setCurrentIndex(lang_id)
        self.lang_combo.currentIndexChanged.connect(self.lang_changed)

        # 图标选择和效果预览合并为一个紧凑区域，稍后放在应用名称之后
        self.preview_group = QGroupBox(lang["preview"][lang_id])
        self.preview_group.setFixedHeight(132)
        preview_layout = QHBoxLayout()
        preview_layout.setContentsMargins(20, 10, 20, 10)
        preview_layout.setSpacing(28)

        self.icon_button = QPushButton(lang["select_icon"][lang_id])
        self.icon_button.clicked.connect(self.select_icon)
        self.icon_button.setFixedSize(220, 38)

        self.icon_preview = QLabel()
        if self._conf["comm"]["icon_file"] and os.path.exists(self._conf["comm"]["icon_file"]):
            img = Image.open(self._conf["comm"]["icon_file"])
            if img.size != (60, 60):
                img = img.resize((60, 60))
            if os.path.exists("model_output"):
                img.save("model_output/icon.png")
            pixmap = QPixmap(self._conf["comm"]["icon_file"])
            self.icon_preview.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.icon_preview.setFixedSize(54, 54)
        self.icon_preview.setAlignment(Qt.AlignCenter)
        self.icon_preview.setStyleSheet("border: 1px solid #666; border-radius: 8px;")

        current_key = LANG_KEYS[lang_id]
        preview_name_text = self._wrap_app_name_text(self._conf["comm"].get(f"app_name_{current_key}", ""))
        self.name_preview = QLabel(self._preview_app_name_text(preview_name_text))
        self.name_preview.setTextFormat(Qt.PlainText)
        self.name_preview.setAlignment(Qt.AlignCenter)
        self.name_preview.setWordWrap(False)
        self.name_preview.setFixedHeight(34)
        self.name_preview.setStyleSheet("font-size: 15px; font-weight: bold;")

        preview_content_layout = QVBoxLayout()
        preview_content_layout.setSpacing(8)
        preview_content_layout.addWidget(self.icon_preview, 0, Qt.AlignHCenter)
        preview_content_layout.addWidget(self.name_preview)

        preview_layout.addStretch(1)
        preview_layout.addWidget(self.icon_button)
        preview_layout.addSpacing(36)
        preview_layout.addLayout(preview_content_layout)
        preview_layout.addStretch(1)
        self.preview_group.setLayout(preview_layout)

        # --- 2. 模式 / 输入数据源 ---
        self.select_mode_group = QGroupBox()
        select_mode_layout = QGridLayout()
        select_mode_layout.setContentsMargins(12, 14, 12, 12)
        select_mode_layout.setHorizontalSpacing(8)
        select_mode_layout.setVerticalSpacing(9)
        select_mode_layout.setColumnMinimumWidth(0, 150)
        select_mode_layout.setColumnStretch(1, 1)

        self.select_mode_label = QLabel(lang["input_source"][lang_id])
        self.select_mode_label.setFixedWidth(150)
        self.select_mode_label.setStyleSheet("font-weight: bold;")
        self.mode_combo = QComboBox()
        self.mode_combo.setFixedHeight(36)
        self.mode_combo.addItems([lang["custom"][lang_id], lang["mindplus_dataset"][lang_id]])
        self.mode_combo.view().setSpacing(2)
        self.mode_combo.view().setStyleSheet("QListView::item { min-height: 36px; padding: 2px 8px; }")
        if self._conf["comm"]["mode"] == "MindPlus":
            self.mode_combo.setCurrentIndex(1)
        else:
            self.mode_combo.setCurrentIndex(0)
        self.mode_combo.currentIndexChanged.connect(self.mode_changed)
        select_mode_layout.addWidget(self.language_label, 0, 0)
        select_mode_layout.addWidget(self.lang_combo, 0, 1)

        self.output_dir_button = QPushButton(lang["output_directory"][lang_id])
        self.output_dir_button.setFixedSize(150, 36)
        self.output_dir_button.setStyleSheet("text-align: left; padding-left: 5px;")
        self.output_dir_button.clicked.connect(self.select_output_dir)
        self.output_dir_label = QLineEdit(self.output_dir)
        self.output_dir_label.setReadOnly(True)
        self.output_dir_label.setFixedHeight(36)
        select_mode_layout.addWidget(self.output_dir_button, 1, 0)
        select_mode_layout.addWidget(self.output_dir_label, 1, 1)

        select_mode_layout.addWidget(self.select_mode_label, 2, 0)
        select_mode_layout.addWidget(self.mode_combo, 2, 1)

        self.select_mode_group.setLayout(select_mode_layout)

        # --- 模型包选择 ---
        self.zip_model_button = QPushButton(lang["select_model_package"][lang_id])
        self.zip_model_button.setFixedSize(150, 36)
        self.zip_model_button.setStyleSheet("text-align: left; padding-left: 5px;")
        self.zip_model_button.clicked.connect(lambda: self.select_zip("model"))
        self.zip_model_label = QLineEdit(self._conf["mindplus_options"]["model_zip"])
        self.zip_model_label.setReadOnly(True)
        self.zip_model_label.setFixedHeight(36)
        select_mode_layout.addWidget(self.zip_model_button, 3, 0)
        select_mode_layout.addWidget(self.zip_model_label, 3, 1)

        # --- 数据包选择 ---
        self.zip_dataset_button = QPushButton(lang["select_dataset_package"][lang_id])
        self.zip_dataset_button.setFixedSize(150, 36)
        self.zip_dataset_button.setStyleSheet("text-align: left; padding-left: 5px;")
        self.zip_dataset_button.clicked.connect(lambda: self.select_zip("dataset"))
        self.zip_dataset_label = QLineEdit(self._conf["mindplus_options"]["dataset_zip"])
        self.zip_dataset_label.setReadOnly(True)
        self.zip_dataset_label.setFixedHeight(36)
        select_mode_layout.addWidget(self.zip_dataset_button, 4, 0)
        select_mode_layout.addWidget(self.zip_dataset_label, 4, 1)

        # --- 自定义目录选择 ---
        self.user_dir_button = QPushButton(lang["custom_directory"][lang_id])
        self.user_dir_button.setFixedSize(150, 36)
        self.user_dir_button.setStyleSheet("text-align: left; padding-left: 5px;")
        self.user_dir_button.clicked.connect(self.select_user_dir)
        self.user_dir_label = QLineEdit(self._conf["user_options"]["user_dir"])
        self.user_dir_label.setReadOnly(True)
        self.user_dir_label.setFixedHeight(36)
        select_mode_layout.addWidget(self.user_dir_button, 5, 0)
        select_mode_layout.addWidget(self.user_dir_label, 5, 1)

        # --- YOLO ONNX 模型选择 ---
        self.user_onnx_button = QPushButton(lang["select_onnx_model"][lang_id])
        self.user_onnx_button.setFixedSize(150, 36)
        self.user_onnx_button.setStyleSheet("text-align: left; padding-left: 5px;")
        self.user_onnx_button.clicked.connect(self.select_user_onnx)
        self.user_onnx_label = QLineEdit(self._conf["user_options"].get("onnx_file", ""))
        self.user_onnx_label.setReadOnly(True)
        self.user_onnx_label.setFixedHeight(36)
        select_mode_layout.addWidget(self.user_onnx_button, 6, 0)
        select_mode_layout.addWidget(self.user_onnx_label, 6, 1)

        # --- YOLO 模型信息（能从 ONNX 元数据读取的项目会锁定） ---
        self.model_series_label = QLabel(lang["model_series"][lang_id])
        self.model_series_label.setFixedWidth(150)
        self.model_series_label.setStyleSheet("font-weight: bold;")
        self.model_series_combo = QComboBox()
        self.model_series_combo.setFixedHeight(36)
        self.model_series_combo.addItems([lang["unknown"][lang_id], "YOLOv8n", "YOLO11n"])
        self.model_series_combo.currentIndexChanged.connect(lambda: self._model_info_changed("series"))
        self.model_series_combo.installEventFilter(self)
        select_mode_layout.addWidget(self.model_series_label, 7, 0)
        select_mode_layout.addWidget(self.model_series_combo, 7, 1)

        self.task_type_label = QLabel(lang["task_type"][lang_id])
        self.task_type_label.setFixedWidth(150)
        self.task_type_label.setStyleSheet("font-weight: bold;")
        self.task_type_combo = QComboBox()
        self.task_type_combo.setFixedHeight(36)
        self.task_type_combo.addItems([
            lang["unknown"][lang_id], lang["task_detect"][lang_id],
            lang["task_classify"][lang_id], lang["task_segment"][lang_id],
        ])
        self.task_type_combo.currentIndexChanged.connect(lambda: self._model_info_changed("task"))
        self.task_type_combo.installEventFilter(self)
        select_mode_layout.addWidget(self.task_type_label, 8, 0)
        select_mode_layout.addWidget(self.task_type_combo, 8, 1)

        self.input_size_label = QLabel(lang["input_size"][lang_id])
        self.input_size_label.setFixedWidth(150)
        self.input_size_label.setStyleSheet("font-weight: bold;")
        self.input_size_combo = QComboBox()
        self.input_size_combo.setFixedHeight(36)
        self.input_size_combo.addItems([lang["unknown"][lang_id], "224 × 224", "320 × 320", "640 × 640"])
        self.input_size_combo.currentIndexChanged.connect(lambda: self._model_info_changed("size"))
        self.input_size_combo.installEventFilter(self)
        select_mode_layout.addWidget(self.input_size_label, 9, 0)
        select_mode_layout.addWidget(self.input_size_combo, 9, 1)

        self.main_layout.addWidget(self.select_mode_group)

        # --- 3. 应用名称 ---
        self.app_name_group = QGroupBox(lang["app_name"][lang_id])
        self.app_name_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        app_name_layout = QVBoxLayout()
        app_name_layout.setSpacing(8)
        app_name_layout.setContentsMargins(12, 14, 12, 12)

        # 当前界面语言的应用名称（与额外语言行同一行，减少纵向空白）
        current_lang_row = QHBoxLayout()
        current_lang_row.setSpacing(8)
        self.app_name_current_lang_label = QLabel(LANG_DISPLAY_NAMES[lang_id])
        self.app_name_current_lang_label.setStyleSheet("font-weight: bold;")
        # 固定标签宽度，与额外语言行左对齐
        label_width = 150
        self.app_name_label_width = label_width
        self.app_name_current_lang_label.setFixedWidth(label_width)
        current_lang_row.addWidget(self.app_name_current_lang_label)
        current_key = LANG_KEYS[lang_id]
        app_name_text = self._conf["comm"].get(f"app_name_{current_key}", "")
        self.app_name_A = QLineEdit()
        self.app_name_A.setText(self._to_app_name_display(app_name_text))
        self.app_name_A.setFixedHeight(36)
        self.app_name_A.setPlaceholderText(lang["app_name_placeholder"][lang_id])
        self.app_name_A.textChanged.connect(self._update_app_name_extra_placeholders)
        self.app_name_A.textChanged.connect(self._update_preview_name)
        self.app_name_A.textChanged.connect(self._update_overall_readiness)
        self.app_name_A.installEventFilter(self)
        current_lang_row.addWidget(self.app_name_A, 1)
        # 与其他语言行的“添加/删除”操作列对齐，使输入框和下拉框等宽
        self.app_name_action_spacer = QWidget()
        self.app_name_action_spacer.setFixedWidth(110)
        current_lang_row.addWidget(self.app_name_action_spacer)
        app_name_layout.addLayout(current_lang_row)

        # 其他语言选择区始终显示，语言输入框仍由用户逐个添加
        self.other_languages_panel = QWidget()
        other_languages_panel_layout = QVBoxLayout()
        other_languages_panel_layout.setContentsMargins(0, 0, 0, 0)
        other_languages_panel_layout.setSpacing(5)

        # 动态添加的其他语言行
        self.app_name_extra_rows_layout = QVBoxLayout()
        self.app_name_extra_rows_layout.setSpacing(5)
        other_languages_panel_layout.addLayout(self.app_name_extra_rows_layout)

        # 添加其他语言
        add_lang_layout = QHBoxLayout()
        add_lang_layout.setSpacing(8)
        self.other_languages_label = QLabel(lang["other_languages"][lang_id])
        self.other_languages_label.setFixedWidth(self.app_name_label_width)
        self.other_languages_label.setStyleSheet("font-weight: bold;")
        add_lang_layout.addWidget(self.other_languages_label)
        self.app_name_lang_combo = QComboBox()
        self.app_name_lang_combo.setFixedHeight(36)
        self.app_name_lang_combo.view().setSpacing(2)
        self.app_name_lang_combo.view().setStyleSheet("QListView::item { min-height: 36px; padding: 2px 8px; }")
        self.add_app_name_lang_btn = QPushButton(lang["add"][lang_id])
        self.add_app_name_lang_btn.setMinimumHeight(36)
        self.add_app_name_lang_btn.setFixedWidth(110)
        self.add_app_name_lang_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.add_app_name_lang_btn.clicked.connect(self.add_app_name_lang)
        add_lang_layout.addWidget(self.app_name_lang_combo, 1)
        add_lang_layout.addWidget(self.add_app_name_lang_btn)
        other_languages_panel_layout.addLayout(add_lang_layout)
        self.other_languages_panel.setLayout(other_languages_panel_layout)
        app_name_layout.addWidget(self.other_languages_panel)

        self.app_name_group.setLayout(app_name_layout)
        self.main_layout.addWidget(self.app_name_group)

        # --- 4. 图标选择与应用效果预览 ---
        self.main_layout.addWidget(self.preview_group)

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
        # 只记录本次界面中主动添加的语言。历史配置仍在 user_added_langs
        # 和 app_name_by_lang 中保留，但不在启动时一次性全部展开。
        self.visible_extra_langs = []

        # 初始化额外语言行和下拉框
        self._sync_from_config()
        self._refresh_lang_combos()

        # --- 5. 阈值设置 ---
        self.threshold_group = QGroupBox(lang["detection_threshold"][lang_id])
        threshold_layout = QHBoxLayout()
        threshold_layout.setContentsMargins(12, 14, 12, 12)
        threshold_layout.setSpacing(12)
        self.threshold_slider = QSlider(Qt.Horizontal)
        # 滑块的每一个整数档位代表 0.02，避免拖动时落在 0.01 的中间值。
        self.threshold_slider.setRange(0, 50)
        self.threshold_slider.setSingleStep(1)
        self.threshold_slider.setPageStep(1)
        initial_threshold_step = round(float(self._conf["comm"]["det_threshold"]) / 0.02)
        self.threshold_slider.setValue(initial_threshold_step)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        self.threshold_label = QLabel(f"{initial_threshold_step * 0.02:.2f}")
        self.threshold_label.setFixedWidth(42)
        self.threshold_label.setAlignment(Qt.AlignCenter)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        self.threshold_group.setLayout(threshold_layout)
        self.main_layout.addWidget(self.threshold_group)

        # --- 6. 开始转换 ---
        self.conversion_status_label = QLabel()
        self.conversion_status_label.setAlignment(Qt.AlignCenter)
        self.conversion_status_label.setWordWrap(True)
        self.conversion_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.conversion_status_label.setMinimumHeight(42)
        status_shadow = QGraphicsDropShadowEffect(self.conversion_status_label)
        status_shadow.setBlurRadius(18)
        status_shadow.setOffset(0, 4)
        status_shadow.setColor(QColor(0, 0, 0, 95))
        self.conversion_status_label.setGraphicsEffect(status_shadow)
        self.conversion_status_label.hide()
        self._set_conversion_status_style(False)
        self.main_layout.addWidget(self.conversion_status_label)

        self.progress_segments_widget = QWidget()
        progress_segments_layout = QHBoxLayout(self.progress_segments_widget)
        progress_segments_layout.setContentsMargins(4, 2, 4, 2)
        progress_segments_layout.setSpacing(5)
        self.progress_segments = []
        for _index in range(7):
            segment = QLabel()
            segment.setFixedHeight(8)
            segment.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            progress_segments_layout.addWidget(segment, 1)
            self.progress_segments.append(segment)
        self.progress_segments_widget.hide()
        self._update_progress_segments()
        self.main_layout.addWidget(self.progress_segments_widget)

        btn_layout = QHBoxLayout()
        self.export_btn = QPushButton(lang["start"][lang_id])
        self.export_btn.clicked.connect(self.export_model)
        self.export_btn.setMinimumSize(240, 44)
        self.export_btn.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.export_btn.setEnabled(False)

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
        btn_layout.setContentsMargins(0, 4, 0, 0)
        self.main_layout.addLayout(btn_layout)

        # 设置主布局
        self.setLayout(self.main_layout)

        # 初始化控件显隐
        self.mode_changed(self.mode_combo.currentIndex())
        QTimer.singleShot(0, self._update_all_app_name_placeholders)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "app_name_A"):
            self._update_all_app_name_placeholders()

    def update_threshold_label(self, value):
        # 每个滑块档位对应 0.02。
        f_value = value * 0.02
        self._conf["comm"]["det_threshold"] = float("{:.2f}".format(f_value))
        self.threshold_label.setText(f"{f_value:.2f}")

    def _threshold_float_value(self):
        return self.threshold_slider.value() * 0.02

    def _work_path(self, name=""):
        base = os.path.abspath(self.work_dir)
        return os.path.join(base, name) if name else base

    def _ensure_work_dirs(self):
        os.makedirs(self._work_path(), exist_ok=True)
        os.chdir(self._work_path())
        for name in ("model_input", "model_output", "dump"):
            os.makedirs(self._work_path(name), exist_ok=True)

    def _reset_work_subdir(self, name):
        allowed = {"model_input", "model_output", "dump", "staging"}
        if name not in allowed:
            raise ValueError(f"Refusing to clear unexpected cache directory: {name}")
        work_root = os.path.realpath(self._work_path())
        target = os.path.realpath(self._work_path(name))
        if os.path.dirname(target) != work_root:
            raise ValueError(f"Unsafe cache path: {target}")
        reset_directory(target)
        return target

    def _set_status(self, text, error=False):
        self.conversion_status_label.setText(text)
        if error:
            self.conversion_status_label.setStyleSheet(
                "QLabel { color: #ffe9ec; "
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
                "stop:0 #3a2229, stop:1 #472831); "
                "border: 1px solid #8f4f5b; border-radius: 10px; "
                "padding: 10px 14px; font-size: 13px; font-weight: 600; }"
            )
        else:
            self._set_conversion_status_style(False)
        self.conversion_status_label.show()
        QTimer.singleShot(0, self._adjust_height_to_content)

    def _update_progress_segments(self):
        """更新七段圆角进度条：已完成点亮，当前步骤使用强调色。"""
        if not hasattr(self, "progress_segments"):
            return
        for step, segment in enumerate(self.progress_segments, 1):
            if step in self._completed_progress_steps:
                background = (
                    "qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                    "stop:0 #258bd2, stop:1 #42b8ed)"
                )
                border = "#56c4ef"
            elif step == self._active_progress_step:
                background = (
                    "qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                    "stop:0 #315873, stop:0.5 #3e789e, stop:1 #315873)"
                )
                border = "#62b9e6"
            else:
                background = "#27333d"
                border = "#34434f"
            segment.setStyleSheet(
                "QLabel {"
                f"background: {background}; border: 1px solid {border};"
                "border-radius: 4px;"
                "}"
            )

    def _progress_step_text(self, step):
        keys = {
            1: "sampling_images",
            2: "preprocessing_images",
            3: "importing_onnx",
            4: "preparing_quantization",
            5: "quantizing_compiling",
            6: "generating_kmodel",
            7: "generating_package",
        }
        return lang[keys[step]][lang_id]

    def _start_progress_step(self, step, current=None, total=None):
        if not 1 <= step <= 7:
            return
        self.progress_segments_widget.show()
        self._active_progress_step = step
        for previous in range(1, step):
            self._completed_progress_steps.add(previous)
        self._update_progress_segments()
        text = self._progress_step_text(step)
        if current is not None and total:
            text = f"{text}  {current} / {total}"
        self._set_status(text)

    def _finish_progress_step(self, step):
        if 1 <= step <= 7:
            self._completed_progress_steps.add(step)
            self._update_progress_segments()

    def _reset_progress_display(self, hide=False):
        self._active_progress_step = 0
        self._completed_progress_steps.clear()
        self._update_progress_segments()
        if hide and hasattr(self, "progress_segments_widget"):
            self.progress_segments_widget.hide()

    def _format_preparation_error(self, error):
        code = error.code
        details = error.details
        localized_keys = {
            "yaml_missing": "yaml_missing_message",
            "yaml_multiple": "yaml_multiple_message",
            "yaml_missing_fields": "yaml_fields_message",
            "onnx_missing": "onnx_missing_message",
            "onnx_multiple": "onnx_multiple_message",
            "unsupported_model": "unsupported_model_message",
            "unsupported_size": "unsupported_size_message",
            "train_directory_missing": "train_missing_message",
            "label_directory_missing": "label_missing_message",
            "classes_uncovered": "classes_uncovered_message",
            "yaml_invalid": "yaml_invalid_message",
            "yaml_path_invalid": "yaml_path_invalid_message",
            "names_invalid": "names_invalid_message",
            "onnx_invalid": "onnx_invalid_message",
            "source_directory_missing": "source_missing_message",
            "train_images_missing": "train_images_empty_message",
            "label_path_unresolved": "label_path_message",
            "label_files_missing": "label_files_empty_message",
            "image_label_pair_missing": "pair_missing_message",
            "class_directory_missing": "class_dir_missing_message",
            "class_images_missing": "class_images_empty_message",
            "model_yaml_missing": "mindplus_yaml_missing_message",
            "model_yaml_multiple": "mindplus_yaml_multiple_message",
            "class_mismatch": "class_mismatch_message",
            "classification_class_count_mismatch": "classification_class_count_mismatch_message",
            "classification_class_count_conflict": "classification_class_count_conflict_message",
            "unsafe_zip": "unsafe_zip_message",
        }
        if code in localized_keys:
            template = lang[localized_keys[code]][lang_id]
            values = dict(details)
            if isinstance(values.get("fields"), list):
                values["fields"] = "、".join(values["fields"])
            if isinstance(values.get("classes"), list):
                values["classes"] = "、".join(values["classes"])
            return template.format(**values)
        if lang_id in (1, 2):
            messages = {
                "yaml_missing": "未找到数据集配置文件，请在所选文件夹中放置一个正确的 .yaml 数据集配置文件。",
                "yaml_multiple": "检测到多个 .yaml 文件，无法确定数据集配置。请仅保留一个正确的数据集配置文件后重试。",
                "yaml_missing_fields": "数据集配置文件缺少以下必要信息：{fields}。",
                "yaml_path_invalid": "数据集配置文件中的 path 和 train 必须是有效路径文本。",
                "yaml_invalid": "无法读取数据集配置文件：\n{path}\n\n{reason}",
                "names_invalid": "数据集配置文件中的 names 信息为空或格式不正确。",
                "onnx_missing": "未找到 ONNX 模型，请在所选文件夹中放入一个正确的 .onnx 文件。",
                "onnx_multiple": "检测到多个 .onnx 文件，无法确定需要转换的模型。请仅保留一个正确的 ONNX 模型后重试。",
                "onnx_invalid": "无法读取 ONNX 模型：\n{path}\n\n{reason}",
                "unsupported_model": "当前仅支持 YOLOv8n 和 YOLO11n，该 ONNX 模型属于不受支持的模型系列：{model}",
                "unsupported_size": "当前仅支持 224 × 224、320 × 320 和 640 × 640，该模型输入尺寸不受支持：{size}",
                "source_directory_missing": "所选数据源文件夹不存在：\n{path}",
                "train_directory_missing": "未找到训练图片目录：\n{path}\n\n请确认 data.yaml 中的 path 和 train 设置正确。",
                "train_images_missing": "训练图片目录中没有可用图片：\n{path}",
                "label_path_unresolved": "无法根据训练图片路径定位标签目录：\n{train_path}\n\n请确保路径中包含 images 目录。",
                "label_directory_missing": "未找到检测/分割训练标签目录：\n{path}\n\n请确认 data.yaml 中的 path 和 train 设置正确，并保证训练图片目录与训练标签目录对应。",
                "label_files_missing": "训练标签目录中没有 .txt 标签文件：\n{path}",
                "image_label_pair_missing": "没有找到同名的训练图片与标签。\n图片目录：{image_path}\n标签目录：{label_path}",
                "class_directory_missing": "未找到类别“{class_name}”对应的分类目录：\n{path}",
                "class_images_missing": "类别“{class_name}”目录中没有可用图片：\n{path}",
                "classes_uncovered": "以下类别没有找到可用校准图片：{classes}",
                "model_yaml_missing": "MindPlus 模型包中缺少模型配置 .yaml 文件。",
                "model_yaml_multiple": "MindPlus 模型包中存在多个 .yaml 文件，无法确定模型配置。",
                "class_mismatch": "MindPlus 模型包与数据集包中的类别信息不一致。",
                "unsafe_zip": "ZIP 中包含不安全的文件路径：{file}",
            }
        else:
            messages = {
                "yaml_missing": "No dataset configuration was found. Keep one correct .yaml dataset configuration file in the selected folder.",
                "yaml_multiple": "Multiple .yaml files were found. Keep only one correct dataset configuration file and try again.",
                "yaml_missing_fields": "The dataset configuration is missing required fields: {fields}.",
                "onnx_missing": "No ONNX model was found. Put one correct .onnx model in the selected folder.",
                "onnx_multiple": "Multiple .onnx files were found. Keep only one correct ONNX model and try again.",
                "unsupported_model": "Only YOLOv8n and YOLO11n are supported. Unsupported model: {model}",
                "unsupported_size": "Only 224 × 224, 320 × 320 and 640 × 640 are supported. Unsupported size: {size}",
                "train_directory_missing": "Training image directory not found:\n{path}",
                "label_directory_missing": "Detection/segmentation label directory not found:\n{path}",
                "classes_uncovered": "No calibration images were found for these classes: {classes}",
            }
        template = messages.get(code, code.replace("_", " "))
        values = dict(details)
        if isinstance(values.get("fields"), list):
            values["fields"] = "、".join(values["fields"])
        if isinstance(values.get("classes"), list):
            values["classes"] = "、".join(values["classes"])
        try:
            return template.format(**values)
        except (KeyError, ValueError):
            return template

    def _invalidate_source(self, clear_model_info=True):
        self._source_info = None
        self._source_ready = False
        self.export_btn.setEnabled(False)
        if not self._conversion_running:
            self._reset_progress_display(hide=True)
        if clear_model_info and hasattr(self, "model_series_combo"):
            self._model_info_updating = True
            for combo in (self.model_series_combo, self.task_type_combo, self.input_size_combo):
                combo.setCurrentIndex(0)
            self._model_info_updating = False
            self._model_info_locked = {"series": False, "task": False, "size": False}
        if hasattr(self, "model_series_combo"):
            for combo in (self.model_series_combo, self.task_type_combo, self.input_size_combo):
                combo.setEnabled(False)

    def _update_overall_readiness(self):
        ready = self._source_ready and bool(self.app_name_A.text().strip())
        self.export_btn.setEnabled(ready and not self._conversion_running)
        if self._conversion_running:
            return
        if self._conf["comm"]["mode"] == "MindPlus":
            if not self._conf["mindplus_options"].get("model_zip", ""):
                self._set_status(lang["select_model_package_first"][lang_id])
                return
            if not self._conf["mindplus_options"].get("dataset_zip", ""):
                self._set_status(lang["select_dataset_package_first"][lang_id])
                return
        else:
            if not self._conf["user_options"].get("user_dir", ""):
                self._set_status(lang["select_dataset_folder_first"][lang_id])
                return
            if not self._conf["user_options"].get("onnx_file", ""):
                self._set_status(lang["select_onnx_model_first"][lang_id])
                return
        if not self._source_ready:
            self._set_status(lang["metadata_incomplete"][lang_id])
        elif not self.app_name_A.text().strip():
            self._set_status(lang["app_name_required_to_start"][lang_id])
        else:
            self._set_status(lang["source_ready"][lang_id])

    def _keep_stage_visible(self, started_at, minimum_ms=750):
        """让很快完成的阶段提示仍有足够时间被用户看见。"""
        remaining = minimum_ms - int((time.monotonic() - started_at) * 1000)
        if remaining <= 0:
            return
        loop = QEventLoop(self)
        QTimer.singleShot(remaining, loop.quit)
        loop.exec_()

    def _show_preparation_error(self, error):
        self._last_preparation_error = error
        message = self._format_preparation_error(error)
        self._set_status(f"{lang['preparation_failed'][lang_id]}\n{message}", True)
        QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], message)

    def _apply_onnx_info(self, info):
        self._model_info_updating = True
        series_indexes = {None: 0, "YOLOv8n": 1, "YOLO11n": 2}
        task_indexes = {None: 0, "detect": 1, "classify": 2, "segment": 3}
        size_indexes = {None: 0, (224, 224): 1, (320, 320): 2, (640, 640): 3}
        self.model_series_combo.setCurrentIndex(series_indexes.get(info.get("series"), 0))
        self.task_type_combo.setCurrentIndex(task_indexes.get(info.get("task"), 0))
        self.input_size_combo.setCurrentIndex(size_indexes.get(info.get("size"), 0))
        self._model_info_updating = False
        self._model_info_locked = {
            "series": info.get("series") is not None,
            "task": info.get("task") is not None,
            "size": info.get("size") is not None,
        }
        for combo in (self.model_series_combo, self.task_type_combo, self.input_size_combo):
            combo.setEnabled(True)

    def _selected_model_info(self):
        series = {1: "YOLOv8n", 2: "YOLO11n"}.get(self.model_series_combo.currentIndex())
        task = {1: "detect", 2: "classify", 3: "segment"}.get(self.task_type_combo.currentIndex())
        size = {1: (224, 224), 2: (320, 320), 3: (640, 640)}.get(self.input_size_combo.currentIndex())
        return series, task, size

    def _finalize_yolo_preparation(self):
        if not self._source_info or self._source_info.get("mode") != "yolo":
            return
        series, task, size = self._selected_model_info()
        if not all((series, task, size)):
            self._source_ready = False
            self.export_btn.setEnabled(False)
            self._set_status(lang["metadata_incomplete"][lang_id])
            return
        try:
            train_dir = self._source_info["train_dir"]
            show_class_count_warning = False
            if task == "classify":
                if self._source_info.get("class_count_conflict"):
                    raise PreparationError("classification_class_count_conflict")
                class_count = self._source_info.get("class_count")
                dataset_count = len(self._source_info["names"])
                if class_count is not None and class_count != dataset_count:
                    raise PreparationError(
                        "classification_class_count_mismatch",
                        model_count=class_count,
                        dataset_count=dataset_count,
                    )
                show_class_count_warning = (
                    class_count is None
                    and not self._source_info.get("class_count_warning_shown", False)
                )
            label_dir = validate_training_source(task, Path(train_dir), self._source_info["names"])
            self._source_info.update({
                "series": series,
                "task": task,
                "size": size,
                "label_dir": str(label_dir) if label_dir else None,
            })
            write_prepared_model_yaml(
                Path(self._work_path("model_input")) / "model.yaml",
                series, task, size, self._source_info["names"], "custom_model",
            )
            self._last_preparation_error = None
            self._source_ready = True
            self._update_overall_readiness()
            if show_class_count_warning:
                self._source_info["class_count_warning_shown"] = True
                QMessageBox.information(
                    self,
                    lang["dialog_warning_title"][lang_id],
                    lang["classification_class_count_unavailable_message"][lang_id],
                )
        except PreparationError as error:
            self._source_ready = False
            self.export_btn.setEnabled(False)
            self._show_preparation_error(error)

    def _model_info_changed(self, field):
        if self._model_info_updating:
            return
        if self._model_info_locked.get(field):
            self._apply_onnx_info(self._source_info or {})
            self._set_status(lang["metadata_locked"][lang_id])
            self._flash_conversion_status()
            return
        self._finalize_yolo_preparation()

    def _prepare_current_source(self):
        self._invalidate_source()
        self._last_preparation_error = None
        try:
            self._ensure_work_dirs()
            self._set_status(lang["preparing_source"][lang_id])
            QApplication.processEvents()
            if self._conf["comm"]["mode"] == "MindPlus":
                model_zip = self._conf["mindplus_options"].get("model_zip", "")
                dataset_zip = self._conf["mindplus_options"].get("dataset_zip", "")
                if not model_zip or not dataset_zip:
                    self._update_overall_readiness()
                    return
                if not os.path.isfile(model_zip):
                    raise PreparationError("source_directory_missing", path=model_zip)
                if not os.path.isfile(dataset_zip):
                    raise PreparationError("source_directory_missing", path=dataset_zip)
                info = inspect_mindplus_source(
                    model_zip, dataset_zip, self._work_path("model_input"), self._work_path("staging")
                )
                self._source_info = info
                self._apply_onnx_info(info)
                self._last_preparation_error = None
                self._source_ready = True
                self._update_overall_readiness()
            else:
                source_dir = self._conf["user_options"].get("user_dir", "")
                onnx_file = self._conf["user_options"].get("onnx_file", "")
                if not source_dir or not onnx_file:
                    self._update_overall_readiness()
                    return
                info = prepare_yolo_source(source_dir, self._work_path("model_input"), onnx_file)
                if info.get("unsupported_series"):
                    raise PreparationError("unsupported_model", model=info["unsupported_series"])
                if info.get("unsupported_size"):
                    raise PreparationError("unsupported_size", size=info["unsupported_size"])
                self._source_info = info
                self._apply_onnx_info(info)
                self._finalize_yolo_preparation()
        except PreparationError as error:
            self._invalidate_source(clear_model_info=False)
            self._show_preparation_error(error)
        except Exception as error:
            self._invalidate_source(clear_model_info=False)
            detail = lang["operation_failed_message"][lang_id].format(reason=error)
            message = f"{lang['preparation_failed'][lang_id]}\n{detail}"
            self._set_status(message, True)
            QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], message)

    def select_zip(self,file_type):
        file, _ = QFileDialog.getOpenFileName(self, lang["select_zip_file"][lang_id], "", "ZIP files (*.zip)")
        if file:
            if file_type == "model":
                self._conf["mindplus_options"]["model_zip"] = file
                self.zip_model_label.setText(file)
            elif file_type == "dataset":
                self._conf["mindplus_options"]["dataset_zip"] = file
                self.zip_dataset_label.setText(file)
            self._prepare_current_source()

    def select_user_dir(self):
        directory = QFileDialog.getExistingDirectory(self, lang["select_custom_directory"][lang_id], "")
        if directory:
            self._conf["user_options"]["user_dir"] = directory
            self.user_dir_label.setText(directory)
            self._prepare_current_source()

    def select_user_onnx(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            lang["select_onnx_model"][lang_id],
            "",
            "ONNX files (*.onnx)",
        )
        if file:
            self._conf["user_options"]["onnx_file"] = file
            self.user_onnx_label.setText(file)
            self._prepare_current_source()

    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            lang["select_output_directory"][lang_id],
            self.output_dir,
        )
        if directory:
            old_output = self.output_dir
            old_work_dir = self.work_dir
            if os.path.normcase(os.path.abspath(directory)) == os.path.normcase(os.path.abspath(old_output)):
                return
            try:
                os.chdir(BASE_DIR)
                if os.path.isdir(old_work_dir):
                    expected = os.path.join(os.path.abspath(old_output), "HUSKYLENS 2 Package Generator Temp Files")
                    if os.path.normcase(os.path.realpath(old_work_dir)) != os.path.normcase(os.path.realpath(expected)):
                        raise ValueError(f"Unsafe old cache path: {old_work_dir}")
                    shutil.rmtree(old_work_dir)
                self.output_dir = directory
                self.work_dir = os.path.join(directory, "HUSKYLENS 2 Package Generator Temp Files")
                self.output_dir_label.setText(directory)
                self._ensure_work_dirs()
                self._prepare_current_source()
            except Exception as error:
                self.output_dir = old_output
                self.work_dir = old_work_dir
                self.output_dir_label.setText(old_output)
                self._ensure_work_dirs()
                message = lang["operation_failed_message"][lang_id].format(reason=error)
                QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], message)

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
            self.icon_preview.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mode_changed(self, index):
        mode_actually_changed = getattr(self, "_last_mode_index", None) != index
        self._last_mode_index = index
        selected_mode = self.mode_combo.itemText(index)
        print("当前模式:", selected_mode)
        if index == 0:
            self._conf["comm"]["mode"] = "User"
            self.zip_model_button.hide()
            self.zip_model_label.hide()
            self.zip_dataset_button.hide()
            self.zip_dataset_label.hide()
            self.user_dir_button.show()
            self.user_dir_label.show()
            self.user_onnx_button.show()
            self.user_onnx_label.show()
            for widget in (
                self.model_series_label, self.model_series_combo,
                self.task_type_label, self.task_type_combo,
                self.input_size_label, self.input_size_combo,
            ):
                widget.show()
        else:
            self._conf["comm"]["mode"] = "MindPlus"
            self.zip_model_button.show()
            self.zip_model_label.show()
            self.zip_dataset_button.show()
            self.zip_dataset_label.show()
            self.user_dir_button.hide()
            self.user_dir_label.hide()
            self.user_onnx_button.hide()
            self.user_onnx_label.hide()
            for widget in (
                self.model_series_label, self.model_series_combo,
                self.task_type_label, self.task_type_combo,
                self.input_size_label, self.input_size_combo,
            ):
                widget.hide()
        self.select_mode_group.updateGeometry()
        self.select_mode_group.adjustSize()
        QTimer.singleShot(0, self._adjust_height_to_content)
        if hasattr(self, "export_btn") and mode_actually_changed:
            self._prepare_current_source()

    def _update_app_name_placeholder(self):
        """根据当前界面语言设置应用名称输入框的占位提示文字。"""
        self.app_name_A.setPlaceholderText(self._app_name_placeholder_for_width(self.app_name_A, lang_id))

    @staticmethod
    def _app_name_placeholder_for_width(line_edit, language_index):
        """完整提示放得下时优先显示，否则使用简短提示。"""
        full_text = lang["app_name_placeholder"][language_index]
        available_width = max(0, line_edit.contentsRect().width() - 12)
        if line_edit.fontMetrics().horizontalAdvance(full_text) <= available_width:
            return full_text
        return lang["app_name_placeholder_short"][language_index]

    def _update_all_app_name_placeholders(self):
        self._update_app_name_placeholder()
        self._update_app_name_extra_placeholders()

    def _update_preview_name(self):
        """预览区只显示应用名称的前两行。"""
        self.name_preview.setText(self._preview_app_name_text(self._to_app_name_value(self.app_name_A.text())))

    @staticmethod
    def _to_app_name_value(display_text):
        """将单行输入框中可见的 \\n 转换为真实换行。"""
        return display_text.replace("\\n", "\n")

    @staticmethod
    def _to_app_name_display(value):
        """将真实换行转换为单行输入框中可见的 \\n。"""
        return value.replace("\n", "\\n")

    @staticmethod
    def _char_display_width(char):
        """计算应用名称字符的显示宽度：半角字符为 1，汉字等全角字符为 2。"""
        if unicodedata.combining(char):
            return 0
        return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1

    @classmethod
    def _wrap_app_name_text(cls, text):
        """按每行 12 个显示宽度单位自动换行，并保留用户手动输入的换行。"""
        wrapped_lines = []
        for source_line in text.split("\n"):
            if not source_line:
                wrapped_lines.append("")
                continue

            current_line = []
            current_width = 0
            for char in source_line:
                char_width = cls._char_display_width(char)
                if current_line and current_width + char_width > 12:
                    wrapped_lines.append("".join(current_line))
                    current_line = [char]
                    current_width = char_width
                else:
                    current_line.append(char)
                    current_width += char_width
            wrapped_lines.append("".join(current_line))
        return "\n".join(wrapped_lines)

    @classmethod
    def _preview_app_name_text(cls, text):
        """格式化名称后仅返回前两行，不修改原始输入内容。"""
        if not text.strip():
            return "App Name"
        wrapped_text = cls._wrap_app_name_text(text)
        return "\n".join(wrapped_text.split("\n")[:2])

    def _update_app_name_extra_placeholders(self):
        """其他语言名称为空时，只显示对应语言的通用输入提示。"""
        for lang_key, widgets in self.app_name_extra_rows.items():
            lang_index = LANG_KEYS.index(lang_key)
            line_edit = widgets["line_edit"]
            line_edit.setPlaceholderText(self._app_name_placeholder_for_width(line_edit, lang_index))

    def _set_button_auto_width(self, button, padding=24):
        """根据按钮文本长度设置最小宽度，保持左右留白一致。"""
        fm = button.fontMetrics()
        text_width = fm.horizontalAdvance(button.text())
        button.setMinimumWidth(text_width + padding)

    def eventFilter(self, obj, event):
        """转换期间拦截界面点击；名称输入框按回车时插入可见换行符。"""
        if self._conversion_running and event.type() == QEvent.MouseButtonPress:
            self._flash_conversion_status()
            return True
        locked_combos = {
            getattr(self, "model_series_combo", None): "series",
            getattr(self, "task_type_combo", None): "task",
            getattr(self, "input_size_combo", None): "size",
        }
        if event.type() == QEvent.MouseButtonPress and obj in locked_combos:
            field = locked_combos[obj]
            if self._model_info_locked.get(field):
                self._set_status(lang["metadata_locked"][lang_id])
                self._flash_conversion_status()
                return True
        if event.type() == QEvent.KeyPress:
            name_edits = [self.app_name_A] + [w["line_edit"] for w in self.app_name_extra_rows.values()]
            if obj in name_edits and event.key() in (Qt.Key_Return, Qt.Key_Enter):
                obj.insert("\\n")
                return True
        return super().eventFilter(obj, event)

    def _set_conversion_status_style(self, highlighted):
        if highlighted:
            style = (
                "QLabel { color: #ffffff; "
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
                "stop:0 #234a64, stop:1 #2d5d7d); "
                "border: 2px solid #62c5f4; border-radius: 10px; "
                "padding: 9px 13px; font-size: 13px; font-weight: 600; }"
            )
        else:
            style = (
                "QLabel { color: #e2f2fb; "
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
                "stop:0 #1c2a35, stop:0.55 #213441, stop:1 #1d303d); "
                "border: 1px solid #38576d; border-radius: 10px; "
                "padding: 10px 14px; font-size: 13px; font-weight: 600; }"
            )
        self.conversion_status_label.setStyleSheet(style)

    def _set_conversion_busy(self, busy):
        """只切换转换期间的界面状态，不影响转换流程。"""
        self._conversion_running = busy
        controls = [
            self.select_mode_group,
            self.app_name_group,
            self.preview_group,
            self.threshold_group,
            self.export_btn,
        ]
        if DEBUG:
            controls.extend([self.save_btn, self.pack_btn])
        for control in controls:
            control.setEnabled(not busy)
        if not busy:
            self.export_btn.setEnabled(self._source_ready and bool(self.app_name_A.text().strip()))

        if busy:
            self.conversion_status_label.setText(lang["converting_please_wait"][lang_id])
            self._set_conversion_status_style(False)
            self.conversion_status_label.show()
        else:
            self._status_flash_timer.stop()
            self._status_flash_step = 0
            self._set_conversion_status_style(False)
            self.conversion_status_label.hide()
        QTimer.singleShot(0, self._adjust_height_to_content)

    def _flash_conversion_status(self):
        """转换期间点击界面时，让底部状态提示短暂闪烁。"""
        self._status_flash_timer.stop()
        self._status_flash_step = 6
        self._set_conversion_status_style(True)
        self.conversion_status_label.repaint()
        self._status_flash_timer.start()

    def _update_status_flash(self):
        self._status_flash_step -= 1
        if self._status_flash_step <= 0:
            self._status_flash_timer.stop()
            self._set_conversion_status_style(False)
            return
        self._set_conversion_status_style(self._status_flash_step % 2 == 0)

    def _sync_from_config(self):
        """根据本次界面中已显示的语言重建输入行。"""
        visible_keys = list(self.visible_extra_langs)
        # 重建界面时只移除控件，不删除已保存的多语言数据。
        for widgets in list(self.app_name_extra_rows.values()):
            layout = widgets["layout"]
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
            self.app_name_extra_rows_layout.removeItem(layout)
            layout.deleteLater()
        self.app_name_extra_rows.clear()
        self.extra_lang_keys.clear()

        current_key = LANG_KEYS[self._current_app_name_lang_id]
        for key in visible_keys:
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
        app_row_layout.setSpacing(8)
        app_label = QLabel(lang_name)
        app_label.setStyleSheet("font-weight: bold;")
        # 固定标签宽度，让所有语言文本框左对齐
        app_label.setFixedWidth(self.app_name_label_width)
        app_line_edit = QLineEdit()
        app_line_edit.setText(self._to_app_name_display(app_text))
        app_line_edit.setFixedHeight(36)
        app_line_edit.setMinimumWidth(220)
        app_line_edit.installEventFilter(self)
        app_line_edit.setPlaceholderText(self._app_name_placeholder_for_width(app_line_edit, lang_index))
        app_delete_btn = QPushButton(lang["delete"][lang_id])
        app_delete_btn.setFixedSize(110, 36)
        app_delete_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        app_delete_btn.clicked.connect(lambda checked, key=lang_key: self._remove_lang_row(key))
        app_row_layout.addWidget(app_label)
        app_row_layout.addWidget(app_line_edit, 1)
        app_row_layout.addWidget(app_delete_btn)
        self.app_name_extra_rows_layout.addLayout(app_row_layout)
        self.app_name_extra_rows[lang_key] = {
            "layout": app_row_layout,
            "label": app_label,
            "line_edit": app_line_edit,
            "delete_btn": app_delete_btn,
        }
        if lang_key not in self.visible_extra_langs:
            self.visible_extra_langs.append(lang_key)
        # 记录用户手动添加的语言
        if lang_key not in self.user_added_langs:
            self.user_added_langs.append(lang_key)
        # 内容变化后自动调整窗口高度，避免其他 Group 被拉伸
        QTimer.singleShot(0, self._update_all_app_name_placeholders)
        QTimer.singleShot(50, self._adjust_height_to_content)

    def _remove_lang_row(self, lang_key):
        """删除应用名称的某一语言输入行。"""
        if lang_key not in self.extra_lang_keys:
            return
        self.extra_lang_keys.discard(lang_key)
        self.app_name_by_lang.pop(lang_key, None)
        if lang_key in self.visible_extra_langs:
            self.visible_extra_langs.remove(lang_key)
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

    def _sync_app_names(self):
        """只同步用户实际输入的名称；缺失语言仅在生成输出配置时补齐。"""
        current_key = LANG_KEYS[self._current_app_name_lang_id]
        self.app_name_by_lang[current_key] = self._to_app_name_value(self.app_name_A.text())
        for key, widgets in self.app_name_extra_rows.items():
            self.app_name_by_lang[key] = self._to_app_name_value(widgets["line_edit"].text())

    def save_conf(self):
        # 同步界面输入并补全所有语言（与导出时 desc.json 使用同一逻辑）
        self._sync_app_names()
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

        # 本次运行的数据只保留在内存中，不写回配置文件，重启后恢复默认。

    def lang_changed(self, index):
        global lang_id
        selected_lang = self.lang_combo.itemText(index)
        print("当前语言:", selected_lang)
        lang_id = self.lang_combo.currentIndex()
        self.setWindowTitle(lang["app_title"][lang_id])
        self.export_btn.setText(lang["start"][lang_id])
        if DEBUG:
            self.save_btn.setText(lang["save_config"][lang_id])
            self.pack_btn.setText(lang["pack_only"][lang_id])
        self.language_label.setText(lang["language"][lang_id])
        self.output_dir_button.setText(lang["output_directory"][lang_id])
        self.select_mode_label.setText(lang["input_source"][lang_id])
        self.user_dir_button.setText(lang["custom_directory"][lang_id])
        self.user_onnx_button.setText(lang["select_onnx_model"][lang_id])
        self.model_series_label.setText(lang["model_series"][lang_id])
        self.task_type_label.setText(lang["task_type"][lang_id])
        self.input_size_label.setText(lang["input_size"][lang_id])
        series_index = self.model_series_combo.currentIndex()
        task_index = self.task_type_combo.currentIndex()
        size_index = self.input_size_combo.currentIndex()
        self._model_info_updating = True
        self.model_series_combo.clear()
        self.model_series_combo.addItems([lang["unknown"][lang_id], "YOLOv8n", "YOLO11n"])
        self.model_series_combo.setCurrentIndex(series_index)
        self.task_type_combo.clear()
        self.task_type_combo.addItems([
            lang["unknown"][lang_id], lang["task_detect"][lang_id],
            lang["task_classify"][lang_id], lang["task_segment"][lang_id],
        ])
        self.task_type_combo.setCurrentIndex(task_index)
        self.input_size_combo.clear()
        self.input_size_combo.addItems([lang["unknown"][lang_id], "224 × 224", "320 × 320", "640 × 640"])
        self.input_size_combo.setCurrentIndex(size_index)
        self._model_info_updating = False
        self.icon_button.setText(lang["select_icon"][lang_id])
        self.preview_group.setTitle(lang["preview"][lang_id])
        self.app_name_group.setTitle(lang["app_name"][lang_id])
        self.threshold_group.setTitle(lang["detection_threshold"][lang_id])
        self.other_languages_label.setText(lang["other_languages"][lang_id])
        self.add_app_name_lang_btn.setText(lang["add"][lang_id])
        for widgets in self.app_name_extra_rows.values():
            widgets["delete_btn"].setText(lang["delete"][lang_id])
        # 切换默认语言：保存旧语言输入，加载新语言输入
        old_key = LANG_KEYS[self._current_app_name_lang_id]
        old_value = self._to_app_name_value(self.app_name_A.text())
        # 当前语言已输入名称时，切换后自动将它保留在“其他语言”区域。
        if old_value.strip():
            self.app_name_by_lang[old_key] = old_value
            if old_key not in self.visible_extra_langs:
                self.visible_extra_langs.append(old_key)
        else:
            self.app_name_by_lang.pop(old_key, None)
        self._current_app_name_lang_id = lang_id
        new_key = LANG_KEYS[lang_id]
        self.app_name_A.setText(self._to_app_name_display(self.app_name_by_lang.get(new_key, "")))
        self.app_name_current_lang_label.setText(LANG_DISPLAY_NAMES[lang_id])
        self._update_app_name_placeholder()

        # 重建额外语言行
        self._sync_from_config()
        self._refresh_lang_combos()

        # 切换语言时保留当前模式，避免 clear 触发 mode_changed 重置
        current_mode_index = self.mode_combo.currentIndex()
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.addItems([lang["custom"][lang_id], lang["mindplus_dataset"][lang_id]])
        self.mode_combo.setCurrentIndex(current_mode_index if current_mode_index >= 0 else 0)
        self.mode_combo.blockSignals(False)
        self.mode_changed(self.mode_combo.currentIndex())

        self.zip_model_button.setText(lang["select_model_package"][lang_id])
        self.zip_dataset_button.setText(lang["select_dataset_package"][lang_id])
        if self._last_preparation_error is not None:
            message = self._format_preparation_error(self._last_preparation_error)
            self._set_status(f"{lang['preparation_failed'][lang_id]}\n{message}", True)
        else:
            self._update_overall_readiness()


    def export_model(self):
        # 用户清理缓存后仍保留已选择的数据源；再次开始时自动重建规范化输入。
        required_inputs = ("best.onnx", "data.yaml", "model.yaml")
        if self._source_info and any(
            not os.path.isfile(self._work_path(os.path.join("model_input", name)))
            for name in required_inputs
        ):
            self._prepare_current_source()
        if not self._source_ready or not self._source_info:
            self._update_overall_readiness()
            self._flash_conversion_status()
            return
        if not self.app_name_A.text().strip():
            print(lang["app_name_cannot_be_empty"][lang_id])
            QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], lang["app_name_cannot_be_empty"][lang_id])
            return

        self._set_conversion_busy(True)
        self._reset_progress_display()
        self._current_stage = "sampling"
        sampling_started_at = time.monotonic()
        self._stage_started_at = sampling_started_at
        self._start_progress_step(1)
        QApplication.processEvents()

        staging_path = self._work_path("staging")
        try:
            self._ensure_work_dirs()
            if self._source_info["mode"] == "mindplus":
                train_dir, label_dir = extract_mindplus_dataset(self._source_info, staging_path)
            else:
                train_dir = Path(self._source_info["train_dir"])
                label_value = self._source_info.get("label_dir")
                label_dir = Path(label_value) if label_value else None

            def sampling_progress(_current, _total):
                QApplication.processEvents()

            result = sample_calibration_images(
                self._source_info["task"], self._source_info["names"],
                train_dir, label_dir, self._work_path("model_input"), sampling_progress,
            )
            print(f"校准图片抽取完成: {result['selected']} 张", flush=True)
        except PreparationError as error:
            self._set_conversion_busy(False)
            message = self._format_preparation_error(error)
            self._set_status(f"{lang['sampling_images'][lang_id]} {message}", True)
            QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], message)
            return
        except Exception as error:
            self._set_conversion_busy(False)
            message = lang["operation_failed_message"][lang_id].format(reason=error)
            self._set_status(f"{lang['sampling_images'][lang_id]} {message}", True)
            QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], message)
            return
        finally:
            if os.path.isdir(staging_path):
                try:
                    shutil.rmtree(staging_path)
                except Exception as error:
                    self._set_conversion_busy(False)
                    message = lang["staging_cleanup_failed"][lang_id].format(path=staging_path, reason=error)
                    self._set_status(message, True)
                    QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], message)
                    return

        self._keep_stage_visible(sampling_started_at)
        self._finish_progress_step(1)
        self._start_progress_step(2)

        self.model_dataset_dir = self._work_path("model_input")
        model_yaml_path = os.path.join(self.model_dataset_dir, "model.yaml")
        print(f"model_yaml_path={model_yaml_path}",flush=True)
        with open(model_yaml_path, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
            print(f"model_config={model_config}",flush=True)
        self.base_model = model_config.get("base_model", "yolov8n")
        yaml_path = os.path.join(self.model_dataset_dir, "data.yaml")
        source_config, name_list = get_name_list_from_data_yaml(yaml_path)
        print(f"source_config={source_config}",flush=True)
        self.dataset_path = os.path.join(self.model_dataset_dir, "images", "train")

        current_key = LANG_KEYS[self._current_app_name_lang_id]
        app_name_current_value = self._to_app_name_value(self.app_name_A.text())
        application = clean_name(self.app_name_A.text())

        # 同步界面输入并补全所有语言：未添加到 user_added_langs 的语言与默认语言一致
        self._sync_app_names()

        conf_data = copy.deepcopy(conf_template)
        conf_data["conf"]["application"] = application
        conf_data["conf"]["model_attach"]["classes"]["zh-CN"] = name_list
        conf_data["conf"]["model_attach"]["classes"]["zh-TW"] = name_list
        conf_data["conf"]["model_attach"]["classes"]["en"] = name_list
        conf_data["conf"]["model_info"][0]["name"] = mindplus_base_model_to_kmodel_base_model[self.base_model][1]
        conf_data["conf"]["model_info"][0]["filename"] = conf_data["conf"]["application"] + ".kmodel"
        conf_data["conf"]["defconfig"]["conf_thres"] = self._threshold_float_value()

        if self.base_model.endswith("-seg"):
            conf_data["conf"]["defconfig"]["det_thres"] = self._threshold_float_value()
            conf_data["conf"]["defconfig"]["nms_thres"] = 0.2
            conf_data["conf"]["defconfig"]["msk_thres"] = conf_data["conf"]["defconfig"]["det_thres"]
        elif self.base_model.endswith("-cls"):
            conf_data["conf"]["defconfig"]["rslt_max_num"] = len(name_list)
        else:
            conf_data["conf"]["defconfig"]["det_thres"] = self._threshold_float_value()
            conf_data["conf"]["defconfig"]["nms_thres"] = 0.2

        try:
            self._reset_work_subdir("model_output")
            self._reset_work_subdir("dump")
        except Exception as error:
            self._set_conversion_busy(False)
            message = lang["operation_failed_message"][lang_id].format(reason=error)
            self._set_status(message, True)
            QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], message)
            return

        model_output_dir = self._work_path("model_output")
        with open(os.path.join(model_output_dir, "conf.json"), "w", encoding="utf-8") as f:
            json.dump(conf_data, f, ensure_ascii=False, indent=4)

        desc_data = copy.deepcopy(desc_template)
        # 以系统语言为默认，缺失的语言用系统语言补齐
        system_key = self._get_system_lang_key()
        system_name = self.app_name_by_lang.get(system_key, app_name_current_value)
        desc_data["desc"]["application_name"] = {
            key: self.app_name_by_lang.get(key) or system_name for key in LANG_KEYS
        }
        # 标题使用对应的应用名称，并将换行替换为空格
        desc_data["desc"]["application_title"] = {
            key: (self.app_name_by_lang.get(key) or system_name).replace("\n", " ")
            for key in LANG_KEYS
        }

        desc_data["desc"]["base_model"] = mindplus_base_model_to_kmodel_base_model[self.base_model][0]

        with open(os.path.join(model_output_dir, "desc.json"), "w", encoding="utf-8") as f:
            json.dump(desc_data, f, ensure_ascii=False, indent=4)
        
        icon_file = self._conf["comm"]["icon_file"]
        if os.path.exists(icon_file):
            shutil.copy(icon_file, os.path.join(model_output_dir, "icon.png"))
        
        # 创建空文件
        open(os.path.join(model_output_dir, f"app.{conf_data['conf']['application']}"), "w").close()

        onnx_path = os.path.join(self.model_dataset_dir, "best.onnx")
        kmodel_path = os.path.join(model_output_dir, conf_data["conf"]["model_info"][0]["filename"])
        output_zip = package_base_name(conf_data["conf"]["application"], self._source_info)

        # 把本次生成的多语言名称等配置同步保存到 app_conf.toml（与 desc.json 同一逻辑）
        self.save_conf()

        # 用独立进程跑转换，避免 nncase 编译占用 GIL 导致 UI 卡死
        self._convert_output = ""
        self._convert_line_buffer = ""
        self._current_stage = "conversion"
        self._stage_started_at = time.monotonic()
        kmodel_conf = os.path.join(RESOURCE_DIR, "kmodel_conf.toml")
        if getattr(sys, "frozen", False):
            # PyInstaller 打包后 sys.executable 是程序自身，用 --run-convertor 进入转换子流程
            convert_args = ["--run-convertor", onnx_path, kmodel_path, self.dataset_path, kmodel_conf, output_zip, self.output_dir]
        else:
            convertor_script = os.path.join(BASE_DIR, "convertor.py")
            convert_args = [convertor_script, onnx_path, kmodel_path, self.dataset_path, kmodel_conf, output_zip, self.output_dir]
        self.process = QProcess(self)
        self.process.setWorkingDirectory(self._work_path())
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_convert_output)
        self.process.finished.connect(self.on_conversion_finished)
        self.process.start(sys.executable, convert_args)
        print("正在转换", flush=True)

    def _on_convert_output(self):
        # 把子进程的输出实时转发到控制台，同时缓存起来用于解析结果路径
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._convert_output += text
        print(text, end="", flush=True)
        self._convert_line_buffer += text
        while "\n" in self._convert_line_buffer:
            line, self._convert_line_buffer = self._convert_line_buffer.split("\n", 1)
            self._handle_convert_progress_line(line.strip())

    def _handle_convert_progress_line(self, line):
        if line == "STAGE=CONVERSION":
            self._current_stage = "conversion"
            return
        if line == "STAGE=PACKAGING":
            self._current_stage = "packaging"
            self._stage_started_at = time.monotonic()
            return
        if line.startswith("PROGRESS_STEP="):
            try:
                self._start_progress_step(int(line.split("=", 1)[1]))
            except ValueError:
                pass
            return
        if line.startswith("PROGRESS_DONE="):
            try:
                self._finish_progress_step(int(line.split("=", 1)[1]))
            except ValueError:
                pass
            return
        if line.startswith("PROGRESS_COUNT="):
            try:
                step, current, total = (int(value) for value in line.split("=", 1)[1].split("|"))
                self._start_progress_step(step, current, total)
            except (ValueError, TypeError):
                pass

    def _clear_task_cache(self):
        """转换成功后自动安全清除本次任务缓存，不再询问用户。"""
        cache_path = os.path.realpath(self._work_path())
        expected = os.path.realpath(os.path.join(self.output_dir, "HUSKYLENS 2 Package Generator Temp Files"))
        try:
            if os.path.normcase(cache_path) != os.path.normcase(expected):
                raise ValueError(f"Unsafe cache path: {cache_path}")
            if os.path.basename(cache_path) != "HUSKYLENS 2 Package Generator Temp Files":
                raise ValueError(f"Unsafe cache directory name: {cache_path}")
            os.chdir(self.output_dir)
            if os.path.isdir(cache_path):
                shutil.rmtree(cache_path)
            # 保留已选择的数据源；再次点击“开始”时自动重新整理缓存。
            self._update_overall_readiness()
        except Exception as error:
            message = lang["operation_failed_message"][lang_id].format(reason=error)
            QMessageBox.warning(self, lang["dialog_warning_title"][lang_id], message)

    def on_conversion_finished(self, exit_code, exit_status):
        if self._current_stage == "packaging":
            self._keep_stage_visible(self._stage_started_at)
        self._set_conversion_busy(False)
        file_path = None
        for line in self._convert_output.splitlines():
            if line.startswith("FINAL_ZIP="):
                file_path = line[len("FINAL_ZIP="):].strip()
        if exit_code == 0 and file_path:
            self._current_stage = "complete"
            self._completed_progress_steps.update(range(1, 8))
            self._active_progress_step = 0
            self._update_progress_segments()
            self._set_status(lang["package_generated"][lang_id])
            QMessageBox.information(
                self,
                lang["conversion_complete_title"][lang_id],
                lang["conversion_complete_message"][lang_id].format(path=file_path),
            )
            print(f"转换完成！文件路径: {file_path}")
            self._clear_task_cache()
        else:
            print(f"转换失败，退出码: {exit_code}")
            failed_text = lang["pack_failed"][lang_id] if self._current_stage == "packaging" else lang["conversion_failed"][lang_id]
            self._set_status(f"{failed_text} (exit code: {exit_code})", True)
            QMessageBox.warning(
                self,
                lang["dialog_warning_title"][lang_id],
                f"{failed_text} (exit code: {exit_code})",
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

        application = clean_name(self.app_name_A.text())
        if self._source_info:
            application = package_base_name(application, self._source_info)
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
    os.makedirs(TEMP_WORK_DIR, exist_ok=True)
    os.chdir(TEMP_WORK_DIR)
    os.makedirs("model_output", exist_ok=True)
    os.makedirs("model_input", exist_ok=True)
    os.makedirs("dump", exist_ok=True)
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(palette=qdarkstyle.DarkPalette) + """
QGroupBox {
    margin-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding-top: 2px;
    padding-left: 14px;
    padding-right: 4px;
}
QPushButton {
    min-height: 28px;
}
QComboBox, QLineEdit {
    min-height: 34px;
    max-height: 34px;
    padding-top: 0px;
    padding-bottom: 0px;
}
""")
    window = ModelExportApp()
    window.setWindowOpacity(1.0)
    window.show()
    sys.exit(app.exec_())
