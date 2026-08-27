# -*- coding: utf-8 -*-
"""数据源检查、规范化与校准图片抽取。"""

from __future__ import annotations

import ast
import os
import random
import re
import shutil
import zipfile
from pathlib import Path

import onnx
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_SERIES = {"YOLOv8n", "YOLO11n"}
SUPPORTED_TASKS = {"detect", "classify", "segment"}
SUPPORTED_SIZES = {(224, 224), (320, 320), (640, 640)}


class PreparationError(Exception):
    def __init__(self, code: str, **details):
        self.code = code
        self.details = details
        super().__init__(code)


def reset_directory(path: str | Path) -> Path:
    target = Path(path).resolve()
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_extract_zip(zip_path: str | Path, output_dir: str | Path) -> Path:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        for entry in archive.infolist():
            target = (destination / entry.filename).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise PreparationError("unsafe_zip", file=entry.filename) from exc
        archive.extractall(destination)
    return destination


def _single_root_file(root: Path, suffix: str, none_code: str, multiple_code: str) -> Path:
    files = sorted(
        (item for item in root.iterdir() if item.is_file() and item.suffix.lower() == suffix),
        key=lambda item: item.name.lower(),
    )
    if not files:
        raise PreparationError(none_code)
    if len(files) > 1:
        raise PreparationError(multiple_code, files=[item.name for item in files])
    return files[0]


def _single_recursive_file(root: Path, suffix: str, none_code: str, multiple_code: str) -> Path:
    files = sorted(
        (item for item in root.rglob(f"*{suffix}") if item.is_file()),
        key=lambda item: str(item).lower(),
    )
    if not files:
        raise PreparationError(none_code)
    if len(files) > 1:
        raise PreparationError(multiple_code, files=[str(item.relative_to(root)) for item in files])
    return files[0]


def _load_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream) or {}
    except Exception as exc:
        raise PreparationError("yaml_invalid", path=str(path), reason=str(exc)) from exc
    if not isinstance(value, dict):
        raise PreparationError("yaml_invalid", path=str(path), reason="root must be a mapping")
    return value


def names_from_config(config: dict, field: str = "names") -> tuple[dict[int, str], list[str]]:
    names = config.get(field)
    if isinstance(names, list):
        mapping = {index: str(name) for index, name in enumerate(names)}
    elif isinstance(names, dict):
        try:
            mapping = {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError) as exc:
            raise PreparationError("names_invalid") from exc
        mapping = dict(sorted(mapping.items()))
    else:
        raise PreparationError("names_invalid")
    if not mapping or any(not value.strip() for value in mapping.values()):
        raise PreparationError("names_invalid")
    return mapping, list(mapping.values())


def validate_dataset_yaml(path: Path) -> tuple[dict, dict[int, str], list[str]]:
    config = _load_yaml(path)
    missing = [field for field in ("train", "names") if config.get(field) in (None, "", [])]
    if missing:
        raise PreparationError("yaml_missing_fields", fields=missing)
    # 用户选择的文件夹就是数据集根目录；省略 path 时以 YAML 所在目录为基准。
    if config.get("path") in (None, ""):
        config["path"] = "."
    if not isinstance(config["path"], str) or not isinstance(config["train"], str):
        raise PreparationError("yaml_path_invalid")
    names, name_list = names_from_config(config)
    return config, names, name_list


def normalize_data_yaml(config: dict, names: dict[int, str], destination: Path) -> None:
    normalized = dict(config)
    normalized["path"] = "./"
    normalized["train"] = "./images/train"
    normalized["names"] = names
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(normalized, stream, allow_unicode=True, sort_keys=False)


def resolve_train_directory(yaml_path: Path, config: dict) -> Path:
    base_value = Path(os.path.expandvars(os.path.expanduser(config["path"])))
    base = base_value if base_value.is_absolute() else yaml_path.parent / base_value
    train_value = Path(os.path.expandvars(os.path.expanduser(config["train"])))
    train = train_value if train_value.is_absolute() else base / train_value
    return train.resolve()


def corresponding_label_directory(train_dir: Path) -> Path:
    parts = list(train_dir.parts)
    image_indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not image_indexes:
        raise PreparationError("label_path_unresolved", train_path=str(train_dir))
    parts[image_indexes[-1]] = "labels"
    return Path(*parts)


def _image_files(directory: Path, recursive: bool = False) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return [item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES]


def _class_directories(train_dir: Path, names: dict[int, str]) -> dict[int, Path]:
    dirs = {item.name: item for item in train_dir.iterdir() if item.is_dir()}
    result = {}
    for class_id, class_name in names.items():
        directory = dirs.get(class_name) or dirs.get(str(class_id))
        if directory is None:
            raise PreparationError("class_directory_missing", class_name=class_name, path=str(train_dir))
        if not _image_files(directory, recursive=True):
            raise PreparationError("class_images_missing", class_name=class_name, path=str(directory))
        result[class_id] = directory
    return result


def validate_training_source(task: str, train_dir: Path, names: dict[int, str]) -> Path | None:
    if not train_dir.is_dir():
        raise PreparationError("train_directory_missing", path=str(train_dir))
    if task == "classify":
        _class_directories(train_dir, names)
        return None

    images = _image_files(train_dir)
    if not images:
        raise PreparationError("train_images_missing", path=str(train_dir))
    label_dir = corresponding_label_directory(train_dir)
    if not label_dir.is_dir():
        raise PreparationError("label_directory_missing", path=str(label_dir))
    labels = [item for item in label_dir.iterdir() if item.is_file() and item.suffix.lower() == ".txt"]
    if not labels:
        raise PreparationError("label_files_missing", path=str(label_dir))
    image_stems = {item.stem for item in images}
    if not any(item.stem in image_stems for item in labels):
        raise PreparationError("image_label_pair_missing", image_path=str(train_dir), label_path=str(label_dir))
    return label_dir


def _parse_size(value) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            numbers = [int(item) for item in re.findall(r"\d+", value)]
            value = numbers
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[-2]), int(value[-1])
        except (TypeError, ValueError):
            return None
    return None


def _tensor_shape(value_info) -> tuple[int | None, ...]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return ()
    result = []
    for dim in tensor_type.shape.dim:
        result.append(int(dim.dim_value) if dim.HasField("dim_value") and dim.dim_value > 0 else None)
    return tuple(result)


def _series_from_texts(texts) -> tuple[str | None, str | None]:
    """只接受文本中明确写出的 YOLO 型号；冲突时不进行猜测。"""
    candidates = set()
    pattern = re.compile(r"(?<![a-z0-9])yolo[\s_-]*v?[\s_-]*(8|11)[\s_-]*([nslmx])\b", re.IGNORECASE)
    for text in texts:
        for match in pattern.finditer(str(text)):
            family = "YOLOv8" if match.group(1) == "8" else "YOLO11"
            candidates.add(f"{family}{match.group(2).lower()}")
    if len(candidates) != 1:
        return None, None
    candidate = next(iter(candidates))
    return (candidate, None) if candidate in SUPPORTED_SERIES else (None, candidate)


def _task_from_texts(texts) -> str | None:
    candidates = set()
    aliases = {
        "detect": "detect", "detection": "detect", "det": "detect",
        "class": "classify", "classify": "classify", "classification": "classify", "cls": "classify",
        "segment": "segment", "segmentation": "segment", "seg": "segment",
    }
    patterns = {
        "classify": r"(?:\bclassif(?:y|ication)\b|(?:^|[-_])cls(?:$|[-_]))",
        "segment": r"(?:\bsegment(?:ation)?\b|(?:^|[-_])seg(?:$|[-_]))",
        "detect": r"(?:\bdetect(?:ion)?\b|(?:^|[-_])det(?:$|[-_]))",
    }
    for text in texts:
        value = str(text).lower()
        normalized = value.strip()
        if normalized in aliases:
            candidates.add(aliases[normalized])
        for task, pattern in patterns.items():
            if re.search(pattern, value):
                candidates.add(task)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _graph_input_size(model) -> tuple[int, int] | None:
    initializer_names = {item.name for item in model.graph.initializer}
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        shape = _tensor_shape(value_info)
        if len(shape) == 4 and shape[-2] and shape[-1]:
            return shape[-2], shape[-1]
    return None


def _graph_task(model) -> str | None:
    """根据明显的输出结构提供辅助证据，不分析不稳定的网络层拓扑。"""
    outputs = [(item.name.lower(), _tensor_shape(item)) for item in model.graph.output]
    if any(re.search(r"mask|proto|segment", name) for name, _shape in outputs):
        return "segment"
    if len(outputs) > 1 and any(len(shape) == 4 and shape[-2] and shape[-1] for _name, shape in outputs):
        return "segment"
    if len(outputs) == 1 and len(outputs[0][1]) == 2:
        return "classify"
    if outputs and all(len(shape) == 3 for _name, shape in outputs):
        return "detect"
    return None


def _classification_class_info(model, metadata_ci: dict) -> tuple[int | None, bool]:
    """返回类别数及是否存在数量冲突，不比较类别名称或顺序。"""
    candidates = set()
    raw_names = metadata_ci.get("names")
    if raw_names not in (None, ""):
        try:
            parsed = ast.literal_eval(str(raw_names)) if isinstance(raw_names, str) else raw_names
            if isinstance(parsed, (list, tuple, dict)) and len(parsed) > 0:
                candidates.add(len(parsed))
        except (SyntaxError, ValueError, TypeError):
            pass
    for output in model.graph.output:
        shape = _tensor_shape(output)
        if len(shape) == 2 and shape[-1]:
            candidates.add(shape[-1])
    if len(candidates) == 1:
        return next(iter(candidates)), False
    return None, len(candidates) > 1


def _directory_contains_image(directory: Path) -> bool:
    try:
        return any(item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES for item in directory.rglob("*"))
    except OSError:
        return False


def _sample_label_files(directory: Path, limit: int = 24) -> list[Path]:
    """蓄水池抽样，目录很大时也不保存全部标签路径。"""
    sample = []
    seen = 0
    try:
        for item in directory.iterdir():
            if not item.is_file() or item.suffix.lower() != ".txt":
                continue
            seen += 1
            if len(sample) < limit:
                sample.append(item)
            else:
                position = random.randrange(seen)
                if position < limit:
                    sample[position] = item
    except OSError:
        return []
    return sample


def _dataset_task(train_dir: Path, names: dict[int, str]) -> str | None:
    """只在数据集结构或少量标签给出单一明确结论时返回任务类型。"""
    try:
        class_dirs = {item.name: item for item in train_dir.iterdir() if item.is_dir()}
    except OSError:
        class_dirs = {}
    expected = [str(name) for name in names.values()]
    if expected and all(name in class_dirs and _directory_contains_image(class_dirs[name]) for name in expected):
        return "classify"

    try:
        label_dir = corresponding_label_directory(train_dir)
    except PreparationError:
        return None
    evidence = set()
    for label in _sample_label_files(label_dir):
        try:
            with label.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    fields = line.strip().split()
                    if not fields:
                        continue
                    if len(fields) == 5:
                        evidence.add("detect")
                    elif len(fields) >= 7 and len(fields) % 2 == 1:
                        evidence.add("segment")
                    break
        except OSError:
            continue
    return next(iter(evidence)) if len(evidence) == 1 else None


def read_onnx_metadata(path: Path, train_dir: Path | None = None,
                       names: dict[int, str] | None = None) -> dict:
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:
        raise PreparationError("onnx_invalid", path=str(path), reason=str(exc)) from exc
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata_ci = {str(key).strip().lower(): value for key, value in metadata.items()}
    text_evidence = list(metadata.keys()) + list(metadata.values()) + [
        model.producer_name, model.producer_version, model.domain,
        model.doc_string, model.graph.doc_string,
    ]
    series, unsupported_series = _series_from_texts(text_evidence)

    strong_task_values = [metadata_ci.get(key) for key in ("task", "head", "model_task", "task_type")]
    strong_task_values = [value for value in strong_task_values if value not in (None, "")]
    task = _task_from_texts(strong_task_values)
    if task is None and not strong_task_values:
        text_task = _task_from_texts(text_evidence)
        graph_task = _graph_task(model)
        dataset_task = _dataset_task(train_dir, names) if train_dir is not None and names else None
        task_evidence = {value for value in (text_task, graph_task, dataset_task) if value is not None}
        task = next(iter(task_evidence)) if len(task_evidence) == 1 else None

    size = None
    for key in ("imgsz", "input_shape", "input_size", "image_size"):
        if key in metadata_ci:
            size = _parse_size(metadata_ci[key])
            if size is not None:
                break
    if size is None:
        size = _graph_input_size(model)
    unsupported_size = size if size and size not in SUPPORTED_SIZES else None
    if unsupported_size:
        size = None

    class_count, class_count_conflict = _classification_class_info(model, metadata_ci)
    return {
        "metadata": metadata,
        "series": series,
        "task": task,
        "size": size,
        "unsupported_series": unsupported_series,
        "unsupported_size": unsupported_size,
        "class_count": class_count,
        "class_count_conflict": class_count_conflict,
    }


def base_model_value(series: str, task: str) -> str:
    prefix = "yolov8n" if series == "YOLOv8n" else "yolo11n"
    if task == "classify":
        return f"{prefix}-cls"
    if task == "segment":
        return f"{prefix}-seg"
    return prefix


def task_from_base_model(base_model: str) -> str:
    if str(base_model).endswith("-cls"):
        return "classify"
    if str(base_model).endswith("-seg"):
        return "segment"
    return "detect"


def series_from_base_model(base_model: str) -> str | None:
    value = str(base_model).lower()
    if value.startswith("yolov8n"):
        return "YOLOv8n"
    if value.startswith("yolo11n"):
        return "YOLO11n"
    return None


def write_model_yaml(destination: Path, series: str, task: str, size: tuple[int, int],
                     names: dict[int, str], description: str) -> dict:
    class FlowList(list):
        pass

    class ModelYamlDumper(yaml.SafeDumper):
        pass

    def represent_flow_list(dumper, value):
        node = dumper.represent_list(value)
        node.flow_style = True
        return node

    ModelYamlDumper.add_representer(FlowList, represent_flow_list)
    aitools_ids = {
        "detect": "ai-tools-detection",
        "classify": "ai-tools-classification",
        "segment": "ai-tools-segmentation",
    }
    config = {
        "aitools_id": aitools_ids[task],
        "aitools_version": "0.0.1",
        "description": description,
        "base_model": base_model_value(series, task),
        "input_shape": FlowList([size[0], size[1]]),
        "labels": names,
    }
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.dump(config, stream, Dumper=ModelYamlDumper, allow_unicode=True, sort_keys=False)
    config["input_shape"] = list(config["input_shape"])
    return config


def prepare_yolo_source(source_dir: str | Path, model_input: str | Path,
                        onnx_file: str | Path | None = None) -> dict:
    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise PreparationError("source_directory_missing", path=str(source))
    if onnx_file:
        onnx_path = Path(onnx_file).resolve()
        if not onnx_path.is_file() or onnx_path.suffix.lower() != ".onnx":
            raise PreparationError("onnx_missing")
    else:
        # 保留旧调用方式的兼容性；界面会单独传入用户选择的 ONNX 文件。
        onnx_path = _single_root_file(source, ".onnx", "onnx_missing", "onnx_multiple")
    preliminary_onnx_info = read_onnx_metadata(onnx_path)

    yaml_files = sorted(
        (item for item in source.iterdir() if item.is_file() and item.suffix.lower() in {".yaml", ".yml"}),
        key=lambda item: item.name.lower(),
    )
    if len(yaml_files) > 1:
        raise PreparationError("yaml_multiple", files=[item.name for item in yaml_files])

    if yaml_files:
        yaml_path = yaml_files[0]
        data_config, names, name_list = validate_dataset_yaml(yaml_path)
        train_dir = resolve_train_directory(yaml_path, data_config)
        onnx_info = read_onnx_metadata(onnx_path, train_dir, names)
    else:
        if preliminary_onnx_info.get("task") != "classify":
            raise PreparationError("yaml_missing")
        yaml_path = None
        train_dir = (source / "train").resolve()
        if not train_dir.is_dir():
            raise PreparationError("train_directory_missing", path=str(train_dir))
        class_directories = sorted(
            (item for item in train_dir.iterdir() if item.is_dir()),
            key=lambda item: item.name.casefold(),
        )
        if not class_directories:
            raise PreparationError("class_directory_missing", class_name="*", path=str(train_dir))
        names = {index: directory.name for index, directory in enumerate(class_directories)}
        name_list = list(names.values())
        data_config = {"path": ".", "train": "train", "names": names}
        onnx_info = preliminary_onnx_info

    target = reset_directory(model_input)
    normalize_data_yaml(data_config, names, target / "data.yaml")
    shutil.copy2(onnx_path, target / "best.onnx")

    return {
        "mode": "yolo",
        "source_dir": str(source),
        "source_yaml": str(yaml_path) if yaml_path else None,
        "source_onnx": str(onnx_path),
        "train_dir": str(train_dir),
        "label_dir": None,
        "names": names,
        "name_list": name_list,
        "data_config": data_config,
        **onnx_info,
    }


def _find_mindplus_model_files(model_root: Path) -> tuple[Path, Path]:
    yaml_path = _single_recursive_file(model_root, ".yaml", "model_yaml_missing", "model_yaml_multiple")
    onnx_path = _single_recursive_file(model_root, ".onnx", "onnx_missing", "onnx_multiple")
    return yaml_path, onnx_path


def _find_named_directory(root: Path, name: str) -> Path | None:
    candidates = [item for item in root.rglob(name) if item.is_dir()]
    candidates.sort(key=lambda item: len(item.parts))
    return candidates[0] if candidates else None


def inspect_mindplus_source(model_zip: str | Path, dataset_zip: str | Path,
                            model_input: str | Path, staging_root: str | Path) -> dict:
    staging = reset_directory(staging_root)
    model_stage = staging / "model"
    dataset_stage = staging / "dataset"
    try:
        safe_extract_zip(model_zip, model_stage)
        safe_extract_zip(dataset_zip, dataset_stage)
        model_yaml, onnx_path = _find_mindplus_model_files(model_stage)
        model_config = _load_yaml(model_yaml)
        names, name_list = names_from_config(model_config, "labels")
        base_model = str(model_config.get("base_model", ""))
        series = series_from_base_model(base_model)
        if series is None:
            raise PreparationError("unsupported_model", model=base_model or "unknown")
        task = task_from_base_model(base_model)
        size = _parse_size(model_config.get("input_shape"))
        if size not in SUPPORTED_SIZES:
            raise PreparationError("unsupported_size", size=size)

        if task == "classify":
            train_dir = _find_named_directory(dataset_stage, "train")
            if train_dir is None:
                raise PreparationError("train_directory_missing", path=str(dataset_stage / "train"))
            data_config = {"path": "./", "train": "./images/train", "names": names}
        else:
            dataset_yamls = sorted(dataset_stage.rglob("*.yaml"))
            if len(dataset_yamls) != 1:
                raise PreparationError("yaml_missing" if not dataset_yamls else "yaml_multiple")
            data_config, dataset_names, _ = validate_dataset_yaml(dataset_yamls[0])
            if dataset_names != names:
                raise PreparationError("class_mismatch")
            train_dir = resolve_train_directory(dataset_yamls[0], data_config)
        label_dir = validate_training_source(task, train_dir, names)

        target = reset_directory(model_input)
        normalize_data_yaml(data_config, names, target / "data.yaml")
        shutil.copy2(onnx_path, target / "best.onnx")
        description = str(model_config.get("description") or "")
        write_model_yaml(target / "model.yaml", series, task, size, names, description)
        return {
            "mode": "mindplus",
            "model_zip": str(Path(model_zip).resolve()),
            "dataset_zip": str(Path(dataset_zip).resolve()),
            "series": series,
            "task": task,
            "size": size,
            "names": names,
            "name_list": name_list,
            "description": description,
            "label_dir": str(label_dir) if label_dir else None,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def extract_mindplus_dataset(info: dict, staging_root: str | Path) -> tuple[Path, Path | None]:
    staging = reset_directory(staging_root)
    dataset_stage = staging / "dataset"
    safe_extract_zip(info["dataset_zip"], dataset_stage)
    if info["task"] == "classify":
        train_dir = _find_named_directory(dataset_stage, "train")
        if train_dir is None:
            raise PreparationError("train_directory_missing", path=str(dataset_stage / "train"))
    else:
        dataset_yamls = sorted(dataset_stage.rglob("*.yaml"))
        if len(dataset_yamls) != 1:
            raise PreparationError("yaml_missing" if not dataset_yamls else "yaml_multiple")
        data_config, _, _ = validate_dataset_yaml(dataset_yamls[0])
        train_dir = resolve_train_directory(dataset_yamls[0], data_config)
    label_dir = validate_training_source(info["task"], train_dir, info["names"])
    return train_dir, label_dir


def _class_quotas(class_ids: list[int]) -> dict[int, int]:
    count = len(class_ids)
    if count == 0:
        raise PreparationError("names_invalid")
    if count > 500:
        return {class_id: 1 for class_id in class_ids}
    base = min(10, 500 // count)
    quotas = {class_id: base for class_id in class_ids}
    remaining = min(500 - base * count, sum(10 - base for _ in class_ids))
    randomized = list(class_ids)
    random.shuffle(randomized)
    index = 0
    while remaining > 0:
        class_id = randomized[index % count]
        if quotas[class_id] < 10:
            quotas[class_id] += 1
            remaining -= 1
        index += 1
    return quotas


def _parse_label_classes(label_path: Path, valid_ids: set[int]) -> set[int]:
    found = set()
    try:
        with label_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                fields = line.strip().split()
                if not fields:
                    continue
                try:
                    class_id = int(float(fields[0]))
                except ValueError:
                    continue
                if class_id in valid_ids:
                    found.add(class_id)
    except OSError:
        return set()
    return found


def sample_calibration_images(task: str, names: dict[int, str], train_dir: str | Path,
                              label_dir: str | Path | None, model_input: str | Path,
                              progress_callback=None) -> dict:
    source_train = Path(train_dir)
    target_root = Path(model_input)
    target_images = target_root / "images" / "train"
    target_labels = target_root / "labels" / "train"
    if target_images.parent.exists():
        shutil.rmtree(target_images.parent)
    if target_labels.parent.exists():
        shutil.rmtree(target_labels.parent)
    target_images.mkdir(parents=True, exist_ok=True)

    quotas = _class_quotas(list(names))
    counts = {class_id: 0 for class_id in names}
    selected_count = 0

    if task == "classify":
        class_dirs = _class_directories(source_train, names)
        for position, (class_id, directory) in enumerate(class_dirs.items(), 1):
            images = _image_files(directory, recursive=True)
            random.shuffle(images)
            chosen = images[:quotas[class_id]]
            class_target = target_images / names[class_id]
            class_target.mkdir(parents=True, exist_ok=True)
            for source in chosen:
                destination = class_target / source.name
                counter = 1
                while destination.exists():
                    destination = class_target / f"{source.stem}_{counter}{source.suffix}"
                    counter += 1
                shutil.copy2(source, destination)
            counts[class_id] = len(chosen)
            selected_count += len(chosen)
            if progress_callback:
                progress_callback(position, len(class_dirs))
    else:
        source_labels = Path(label_dir) if label_dir else corresponding_label_directory(source_train)
        target_labels.mkdir(parents=True, exist_ok=True)
        images_by_stem = {item.stem: item for item in _image_files(source_train)}
        labels = [item for item in source_labels.iterdir() if item.is_file() and item.suffix.lower() == ".txt"]
        random.shuffle(labels)
        valid_ids = set(names)
        for position, label in enumerate(labels, 1):
            class_ids = _parse_label_classes(label, valid_ids)
            image = images_by_stem.get(label.stem)
            needed = image is not None and any(counts[class_id] < quotas[class_id] for class_id in class_ids)
            if needed:
                shutil.copy2(image, target_images / image.name)
                shutil.copy2(label, target_labels / label.name)
                selected_count += 1
                for class_id in class_ids:
                    if counts[class_id] < quotas[class_id]:
                        counts[class_id] += 1
            if progress_callback and position % 50 == 0:
                progress_callback(position, len(labels))
            if all(counts[class_id] >= quotas[class_id] for class_id in counts):
                break

    uncovered = [names[class_id] for class_id, count in counts.items() if count == 0]
    if uncovered:
        raise PreparationError("classes_uncovered", classes=uncovered)
    return {"selected": selected_count, "counts": counts, "quotas": quotas}
