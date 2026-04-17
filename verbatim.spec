# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Verbatim."""
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
project_dir = os.path.dirname(os.path.abspath(SPECPATH))
# Use the venv that uv actually created for this project.
import site, sysconfig
venv_sp = sysconfig.get_path("purelib")

# Collect ALL submodules + data for packages with deep dynamic imports.
_extra_datas = []
_extra_binaries = []
_extra_hiddenimports = []
for pkg in ("pyannote", "parakeet_mlx", "mlx", "asteroid_filterbanks",
            "speechbrain", "pytorch_metric_learning", "lightning_fabric",
            "pytorch_lightning"):
    try:
        d, b, h = collect_all(pkg)
        _extra_datas += d
        _extra_binaries += b
        _extra_hiddenimports += h
    except Exception:
        pass

a = Analysis(
    ["gui.py"],
    pathex=[project_dir],
    binaries=[
        # MLX native libraries (Metal shaders + dylib)
        (os.path.join(venv_sp, "mlx", "lib", "libmlx.dylib"), "mlx/lib"),
        (os.path.join(venv_sp, "mlx", "lib", "mlx.metallib"), "mlx/lib"),
        # Static ffmpeg binary (no Homebrew deps)
        ("vendor/ffmpeg", "."),
    ] + _extra_binaries,
    datas=[
        ("web", "web"),
        ("backends.py", "."),
        ("transcribe.py", "."),
        ("logging_setup.py", "."),
    ] + _extra_datas,
    hiddenimports=[
        "backends",
        "transcribe",
        "logging_setup",
        "parakeet_mlx",
        "parakeet_mlx.audio",
        "parakeet_mlx.conformer",
        "parakeet_mlx.tokenizer",
        "parakeet_mlx.alignment",
        "parakeet_mlx.ctc",
        "parakeet_mlx.rnnt",
        "parakeet_mlx.attention",
        "mlx",
        "mlx._reprlib_fix",
        "mlx.core",
        "mlx.extension",
        "mlx.nn",
        "mlx.nn.init",
        "mlx.nn.layers",
        "mlx.nn.layers.activations",
        "mlx.nn.layers.base",
        "mlx.nn.layers.containers",
        "mlx.nn.layers.convolution",
        "mlx.nn.layers.dropout",
        "mlx.nn.layers.embedding",
        "mlx.nn.layers.linear",
        "mlx.nn.layers.normalization",
        "mlx.nn.layers.positional_encoding",
        "mlx.nn.layers.quantized",
        "mlx.nn.layers.recurrent",
        "mlx.nn.layers.transformer",
        "mlx.nn.utils",
        "mlx.optimizers",
        "mlx.utils",
        "pyannote.audio",
        "pyannote.audio.pipelines",
        "pyannote.audio.pipelines.speaker_diarization",
        "pyannote.core",
        "pyannote.pipeline",
        "torch",
        "torchaudio",
        "soundfile",
        "webview",
        "numpy",
        "scipy",
        "sklearn",
        "sklearn.cluster",
        "huggingface_hub",
    ] + _extra_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["toga", "tkinter", "matplotlib", "IPython", "jupyter"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Verbatim",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch="arm64",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Verbatim",
)
app = BUNDLE(
    coll,
    name="Verbatim.app",
    icon='assets/Verbatim.icns',
    bundle_identifier="com.sls.verbatim",
    info_plist={
        "CFBundleName": "Verbatim",
        "CFBundleDisplayName": "Verbatim",
        "CFBundleVersion": "0.3.0",
        "CFBundleShortVersionString": "0.3.0",
        "LSMinimumSystemVersion": "14.0",
    },
)
