# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_data_files

nncase_datas, nncase_binaries, nncase_hiddenimports = collect_all("nncase")
qdarkstyle_datas = collect_data_files("qdarkstyle")

unused_binary_parts = (
    "PIL\\_avif",
    "PIL\\_imagingcms",
    "PIL\\_imagingmath",
    "PIL\\_imagingtk",
    "PIL\\_webp",
    "PyQt5\\Qt5\\bin\\Qt5DBus.dll",
    "PyQt5\\Qt5\\bin\\Qt5Network.dll",
    "PyQt5\\Qt5\\bin\\Qt5Qml.dll",
    "PyQt5\\Qt5\\bin\\Qt5QmlModels.dll",
    "PyQt5\\Qt5\\bin\\Qt5Quick.dll",
    "PyQt5\\Qt5\\bin\\Qt5WebSockets.dll",
    "PyQt5\\Qt5\\bin\\d3dcompiler_47.dll",
    "PyQt5\\Qt5\\bin\\libEGL.dll",
    "PyQt5\\Qt5\\bin\\libGLESv2.dll",
    "PyQt5\\Qt5\\plugins\\generic\\",
    "PyQt5\\Qt5\\plugins\\platforms\\qminimal.dll",
    "PyQt5\\Qt5\\plugins\\platforms\\qoffscreen.dll",
    "PyQt5\\Qt5\\plugins\\platforms\\qwebgl.dll",
    "PyQt5\\Qt5\\plugins\\platformthemes\\",
    "PyQt5\\Qt5\\translations\\",
    "cv2\\opencv_videoio_ffmpeg",
)


def keep_required_binary(entry):
    destination = entry[0]
    return not any(part.lower() in destination.lower() for part in unused_binary_parts)

analysis = Analysis(
    ["app.py"],
    pathex=[],
    binaries=nncase_binaries,
    datas=[
        ("app_conf.toml", "."),
        ("kmodel_conf.toml", "."),
        ("icon.png", "."),
        ("exe_icon.png", "."),
    ] + nncase_datas + qdarkstyle_datas,
    hiddenimports=["convertor"] + nncase_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PIL.AvifImagePlugin",
        "PIL.FpxImagePlugin",
        "PIL.ImageCms",
        "PIL.ImageMath",
        "PIL.ImageTk",
        "PIL.MicImagePlugin",
        "PIL.MspImagePlugin",
        "PIL.WebPImagePlugin",
        "PyQt5.QtDBus",
        "PyQt5.QtNetwork",
        "PyQt5.QtQml",
        "PyQt5.QtQuick",
        "PyQt5.QtWebSockets",
    ],
    noarchive=False,
    optimize=0,
)

analysis.binaries = [entry for entry in analysis.binaries if keep_required_binary(entry)]

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="HUSKYLENS2_Package_Generator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["exe_icon.png"],
)
