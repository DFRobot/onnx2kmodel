#!/usr/bin/env python3

import subprocess
import argparse
import nncase
import tomlkit
import sys
import os
import cv2
import numpy as np
from pathlib import Path

swapRB = False
preprocess = False

input_type = np.uint8

templs_shape = 640


# setup env
result = subprocess.run(["pip", "show", "nncase"], capture_output=True)
line_break = "\n"
if sys.platform == "win32":
    line_break = "\r\n"
location_s = [i for i in result.stdout.decode().split(
    line_break) if i.startswith("Location:")]
location = location_s[0].split(": ")[1]
if "PATH" in os.environ:
    os.environ["PATH"] += os.pathsep + location
else:
    os.environ["PATH"] = location
os.environ["NNCASE_PLUGIN_PATH"] = location


class Convertor(nncase.Compiler):
    kmodel: str

    def __init__(self, model: str, kmodel: str, conf: str, calib: list):
        _conf: map
        with open(conf, 'r') as f:
            _conf = tomlkit.parse(f.read())
            
        super().__init__(self._set_cpl_opt(_conf))
        with open(model, 'rb') as f:
            _model = Path(model)
            if _model.suffix == ".onnx":
                self.import_onnx(f.read(), nncase.ImportOptions())
            else:
                assert False, print('not support model type')
        self.use_ptq(self._set_ptq_opt(_conf, calib))
        self.kmodel = kmodel

    def convert(self):
        self.compile()
        with open(self.kmodel, 'wb') as f:
            f.write(self.gencode_tobytes())

    def _set_cpl_opt(self, conf: map):
        compile_options = nncase.CompileOptions()
        compile_options.target = conf['compile_options']['target']
        compile_options.dump_ir = conf['compile_options']['dump_ir']
        compile_options.dump_asm = conf['compile_options']['dump_asm']
        compile_options.dump_dir = conf['compile_options']['dump_dir']
        compile_options.input_file = conf['compile_options']['input_file']
        compile_options.preprocess = conf['compile_options']['preprocess']
        compile_options.input_type = conf['compile_options']['input_type']
        compile_options.input_shape = conf['compile_options']['input_shape']
        compile_options.input_range = conf['compile_options']['input_range']
        compile_options.input_layout = conf['compile_options']['input_layout']
        compile_options.swapRB = conf['compile_options']['swapRB']
        compile_options.mean = conf['compile_options']['mean']
        compile_options.std = conf['compile_options']['std']
        compile_options.letterbox_value = conf['compile_options']['letterbox_value']
        compile_options.output_layout = conf['compile_options']['output_layout']

        return compile_options

    def _set_ptq_opt(self, conf: map, calib: list):
        ptq_options = nncase.PTQTensorOptions()
        ptq_options.calibrate_method = conf['ptq_options']['calibrate_method']
        ptq_options.finetune_weights_method = conf['ptq_options']['finetune_weights_method']
        ptq_options.quant_type = conf['ptq_options']['quant_type']
        ptq_options.w_quant_type = conf['ptq_options']['w_quant_type']
        ptq_options.dump_quant_error = conf['ptq_options']['dump_quant_error']
        ptq_options.dump_quant_error_symmetric_for_signed = conf[
            'ptq_options']['dump_quant_error_symmetric_for_signed']
        ptq_options.quant_scheme = conf['ptq_options']['quant_scheme']
        ptq_options.quant_scheme_strict_mode = conf['ptq_options']['quant_scheme_strict_mode']
        ptq_options.export_quant_scheme = conf['ptq_options']['export_quant_scheme']
        ptq_options.export_weight_range_by_channel = conf[
            'ptq_options']['export_weight_range_by_channel']

        ptq_options.samples_count = len(calib[0])
        ptq_options.set_tensor_data(calib)

        return ptq_options


def padding(img):
    h, w = img.shape[:2]

    scale = max(w, h)
    pad_top    = (scale - h) // 2
    pad_bottom = (scale - h) - pad_top
    pad_left   = (scale - w) // 2
    pad_right  = (scale - w) - pad_left

    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img


def process_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = padding(img)
    #print(img.shape)
    img = cv2.resize(img, (templs_shape, templs_shape))
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

'''
def gen(_dir):
    path = Path(_dir)
    files = [f.name for f in path.rglob('*') if f.is_file()]
    for f in files:
        img_path = os.path.join(_dir, f)
        #print(img_path)
        templ = cv2.imread(img_path)
        templ = process_img(templ)
        yield templ
'''

def gen(_dir):
    path = Path(_dir)

    for f in path.rglob('*'):
        if not f.is_file():
            continue
        
        img_path = str(f)   # 关键：不要 f.name

        data = np.fromfile(img_path, dtype=np.uint8)
        templ = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if templ is None:
            print(f"[WARN] read failed: {img_path}")
            continue

        templ = process_img(templ)
        yield templ

def make(onnx_file, kmodel_file, dataset, toml_file, input_shape):
    global templs_shape
    templs_shape = input_shape[2]
    calib = []
    for t in gen(dataset):
        calib.append(t)

    npcalib = np.array(calib).astype(np.uint8)

    #print("calib shape", npcalib.shape)
    print("toml_file  ",toml_file)
    with open(toml_file, 'r', encoding='utf-8') as f:
            conf = tomlkit.parse(f.read())
            print(conf)
            conf['compile_options']['input_shape'] = input_shape
            print(conf)
    with open(toml_file, 'w', encoding='utf-8') as f:
            f.write(tomlkit.dumps(conf))
    c = Convertor(onnx_file, kmodel_file, toml_file, [npcalib])
    c.convert()


def _zip_with_md5(source_dir="model_output/", zip_dir="./", base_name="app"):
    import zipfile
    import hashlib
    import shutil
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
    print(f"pack done: {final_zip_path}", flush=True)
    return final_zip_path


def _get_input_shape(onnx_path):
    import onnx
    model = onnx.load(onnx_path)
    input_shapes = {}
    for tensor in model.graph.input:
        shape = []
        tensor_type = tensor.type.tensor_type
        if tensor_type.HasField('shape'):
            for dim in tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                else:
                    shape.append(None)
        input_shapes[tensor.name] = shape
    return input_shapes


def main(argv=None):
    """命令行入口：供 python convertor.py 直接调用，也供 PyInstaller 打包后的
    主程序以 --run-convertor 参数在子进程中调用。返回进程退出码。"""
    # 子进程的输出由 app.py 按 UTF-8 解码；Windows 中文系统下 stdout 默认是 GBK，
    # 统一强制为 UTF-8，否则 FINAL_ZIP 里的中文路径会变成乱码
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Convert ONNX to KModel and package the output zip.")
    parser.add_argument("onnx_file")
    parser.add_argument("kmodel_file")
    parser.add_argument("dataset")
    parser.add_argument("toml_file")
    parser.add_argument("output_zip")
    args = parser.parse_args(argv)

    shapes = _get_input_shape(args.onnx_file)
    if len(shapes) != 1:
        print(f"[ERROR] unsupported input count: {len(shapes)}", flush=True)
        return 1
    shape = list(shapes.values())[0]
    print(f"input_shape={shape}", flush=True)

    make(args.onnx_file, args.kmodel_file, args.dataset, args.toml_file, shape)

    final_zip_path = _zip_with_md5(base_name=args.output_zip)
    print(f"FINAL_ZIP={final_zip_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
