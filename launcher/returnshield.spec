# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the ReturnShield supervisor launcher.
#
# The launcher is deliberately stdlib-only: freezing Streamlit + sklearn + shap
# + xgboost into one binary is fragile and enormous. The exe is a small
# supervisor; the application ships as source + assets beside it and runs on
# the machine's Python.

a = Analysis(
    ['returnshield_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'pytest', 'IPython'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ReturnShield',
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
