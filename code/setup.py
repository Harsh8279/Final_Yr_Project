import sys
from cx_Freeze import setup, Executable
# from PyInstaller.utils.hooks import is_module_satisfies

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
        'packages': ["os","sys","ctypes"],
        'include_files': ['res'],
        'include_msvcr': True
}

# GUI applications require a different base on Windows (the default is for
# a console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name = "Mst Detector",
    version = "0.1",
    description = "A Detector",
    author = "Team",
    options = {"build_exe": build_exe_options},
    executables = [Executable("detection.py", target_name="MST-Detector.exe", shortcut_name="MST-Detector",shortcut_dir="DesktopFolder", base=base)]
)