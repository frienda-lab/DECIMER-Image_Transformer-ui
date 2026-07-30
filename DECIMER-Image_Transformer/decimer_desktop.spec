# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path(SPECPATH)
datas = [
    (str(project_root / "models"), "models"),
    (str(project_root / "DECIMER_SEGMENTATION_LICENSE"), "."),
    (str(project_root / "LICENSE"), "."),
    (str(project_root / "UI_README_zh.md"), "."),
]
binaries = []
hiddenimports = [
    "efficientnet.tfkeras",
    "pillow_heif.HeifImagePlugin",
    "keras.src.legacy.saving.legacy_h5_format",
]

a = Analysis(
    ["decimer_desktop.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "notebook",
        "jupyter",
        "tkinter",
        "torch",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DECIMER Desktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DECIMER Desktop",
)
