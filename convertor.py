#!/usr/bin/env python3

import subprocess
import argparse
import nncase
import toml
import sys
import os
import cv2
import numpy as np
from pathlib import Path

swapRB = False
preprocess = False

input_type = np.uint8

templs_shape = 320


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
            _conf = toml.load(f)

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
    img = cv2.resize(img, (320, 320))
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def gen(_dir):
    path = Path(_dir)
    files = [f.name for f in path.rglob('*') if f.is_file()]
    for f in files:
        img_path = os.path.join(_dir, f)
        #print(img_path)
        templ = cv2.imread(img_path)
        templ = process_img(templ)
        yield templ


def make(onnx_file, kmodel_file, dataset, toml_file):
    calib = []
    for t in gen(dataset):
        calib.append(t)

    npcalib = np.array(calib).astype(np.uint8)

    #print("calib shape", npcalib.shape)

    c = Convertor(onnx_file, kmodel_file, toml_file, [npcalib])
    c.convert()
