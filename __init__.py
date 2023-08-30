import os
import sys
import subprocess
import shlex
import filelock

pkg_dir = os.path.dirname(__file__)

with filelock.FileLock(os.path.join(pkg_dir, f".lock")):
    try:
        from custom_op.decode_and_crop_jpeg import decode_and_crop_jpeg
    except ImportError:
        install_cmd = f"{sys.executable} setup.py build_ext --inplace"
        subprocess.run(shlex.split(install_cmd), capture_output=True, cwd=pkg_dir)
        from decode_and_crop_jpeg import decode_and_crop_jpeg
