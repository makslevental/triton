import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple


def get_version():
    if os.environ.get("GITHUB_ACTIONS") == "true":
        commit_msg = subprocess.check_output(
            ["git", "log", "--oneline", "-n", "1"], text=True
        )
        base_commit = re.findall(r"Merge \w+ into (\w+)", commit_msg)
        assert base_commit, "couldn't find base commit"
        latest_commit = base_commit[0][:8]
    else:
        latest_commit = subprocess.check_output(
            ["git", "merge-base", "origin/main", "HEAD"], text=True
        ).strip()[:8]
    now = datetime.now()
    now = os.environ.get(
        "DATETIME", f"{now.year}{now.month:02}{now.day:02}{now.hour:02}"
    )
    # in order for the wheels to be ordered chronologically
    # include the epoch seconds as a portion of the version
    return f"{now}+{latest_commit}"


def get_base_dir():
    return Path(__file__).parent.parent.parent


def get_llvm_package_info():
    system = platform.system()
    try:
        arch = {"x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}[
            platform.machine()
        ]
    except KeyError:
        arch = platform.machine()
    if system == "Darwin":
        system_suffix = f"macos-{arch}"
    elif system == "Linux":
        if arch == "arm64":
            system_suffix = "ubuntu-arm64"
        elif arch == "x64":
            vglibc = tuple(map(int, platform.libc_ver()[1].split(".")))
            vglibc = vglibc[0] * 100 + vglibc[1]
            if vglibc > 228:
                # Ubuntu 24 LTS (v2.39)
                # Ubuntu 22 LTS (v2.35)
                # Ubuntu 20 LTS (v2.31)
                system_suffix = "ubuntu-x64"
            elif vglibc > 217:
                # Manylinux_2.28 (v2.28)
                # AlmaLinux 8 (v2.28)
                system_suffix = "almalinux-x64"
            else:
                # Manylinux_2014 (v2.17)
                # CentOS 7 (v2.17)
                system_suffix = "centos-x64"
        else:
            raise Exception(
                f"LLVM pre-compiled image is not available for {system}-{arch}. Proceeding with user-configured LLVM from source build."
            )
    else:
        raise Exception(
            f"LLVM pre-compiled image is not available for {system}-{arch}. Proceeding with user-configured LLVM from source build."
        )
    # use_assert_enabled_llvm = check_env_flag("TRITON_USE_ASSERT_ENABLED_LLVM", "False")
    # release_suffix = "assert" if use_assert_enabled_llvm else "release"
    llvm_hash_path = os.path.join(get_base_dir(), "cmake", "llvm-hash.txt")
    with open(llvm_hash_path, "r") as llvm_hash_file:
        rev = llvm_hash_file.read(8)
    name = f"llvm-{rev}-{system_suffix}"
    return f"https://oaitriton.blob.core.windows.net/public/llvm-builds/{name}.tar.gz"


if len(sys.argv) > 1 and sys.argv[1] == "--llvm-url":
    print(get_llvm_package_info())
    exit()


# https://stackoverflow.com/a/36693250/9045206
def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


extra_files = package_files("triton_mlir/_mlir_libs")

from setuptools import setup, Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True


setup(
    name="triton-mlir",
    package_data={"": extra_files},
    distclass=BinaryDistribution,
    version=get_version(),
    install_requires=[
        "triton-mlir-python-extras @ git+https://github.com/makslevental/mlir-python-extras@makslevental/triton_mlir"
    ],
)
