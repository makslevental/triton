import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple


def get_version():
    latest_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    now = datetime.now()
    now = os.environ.get(
        "DATETIME", f"{now.year}{now.month:02}{now.day:02}{now.hour:02}"
    )
    # in order for the wheels to be ordered chronologically
    # include the epoch seconds as a portion of the version
    return f"{now}+{latest_commit}"


class Package(NamedTuple):
    package: str
    name: str
    url: str
    include_flag: str
    lib_flag: str
    syspath_var_name: str


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
            print(
                f"LLVM pre-compiled image is not available for {system}-{arch}. Proceeding with user-configured LLVM from source build."
            )
            return Package(
                "llvm",
                "LLVM-C.lib",
                "",
                "LLVM_INCLUDE_DIRS",
                "LLVM_LIBRARY_DIR",
                "LLVM_SYSPATH",
            )
    else:
        print(
            f"LLVM pre-compiled image is not available for {system}-{arch}. Proceeding with user-configured LLVM from source build."
        )
        return Package(
            "llvm",
            "LLVM-C.lib",
            "",
            "LLVM_INCLUDE_DIRS",
            "LLVM_LIBRARY_DIR",
            "LLVM_SYSPATH",
        )
    # use_assert_enabled_llvm = check_env_flag("TRITON_USE_ASSERT_ENABLED_LLVM", "False")
    # release_suffix = "assert" if use_assert_enabled_llvm else "release"
    llvm_hash_path = os.path.join(get_base_dir(), "cmake", "llvm-hash.txt")
    with open(llvm_hash_path, "r") as llvm_hash_file:
        rev = llvm_hash_file.read(8)
    name = f"llvm-{rev}-{system_suffix}"
    url = f"https://oaitriton.blob.core.windows.net/public/llvm-builds/{name}.tar.gz"
    return Package(
        "llvm", name, url, "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH"
    )


if len(sys.argv) > 1 and sys.argv[1] == "--llvm-url":
    print(get_llvm_package_info().url)
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
