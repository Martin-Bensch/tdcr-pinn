import os
from setuptools import setup, find_packages
from sys import platform

from pybind11.setup_helpers import Pybind11Extension, build_ext


def _split_env_paths(var_name):
    value = os.environ.get(var_name, "")
    return [p for p in value.split(":") if p]


if platform == "linux" or platform == "linux2":
    extension_headers = [
        "/usr/include/eigen3",
        "/usr/include/eigen",
        "/usr/include",
    ] + _split_env_paths("EIGEN_INCLUDE_DIRS")

    extension_libraries_dirs = [
        "/usr/lib",
        "/usr/lib/x86_64-linux-gnu",
    ] + _split_env_paths("GSL_LIBRARY_DIRS")
    extension_libraries_linking = ["gsl", "gslcblas"]
    extra_linking_args = ["-Wl,--no-as-needed"]
elif platform == "darwin":
    extension_headers = [
        "/opt/homebrew/include/eigen3",
        "/usr/local/include/eigen3",
        "/usr/local/include",
    ] + _split_env_paths("EIGEN_INCLUDE_DIRS")

    extension_libraries_dirs = [
        "/opt/homebrew/lib",
        "/usr/local/lib",
    ] + _split_env_paths("GSL_LIBRARY_DIRS")
    extension_libraries_linking = ["gsl", "gslcblas"]
    extra_linking_args = []
elif platform == "win32":
    raise NotImplementedError("Windows is currently not supported.")
else:
    raise NotImplementedError(f"Unsupported platform: {platform}")

__version__ = "1.10.0"

ext_modules = [
    Pybind11Extension(name="tdrpyb",
        sources=
            ["src/pytdcrsv/binding_code/bind_to_python.cpp",
             "src/pytdcrsv/binding_code/tendondrivenrobot_pyb.cpp",
             "src/pytdcrsv/tdcr_modeling/src/tendondrivenrobot.cpp",
             "src/pytdcrsv/tdcr_modeling/src/cosseratrodmodel.cpp",
             "src/pytdcrsv/tdcr_modeling/src"
             "/constantcurvaturemodel.cpp",
             "src/pytdcrsv/tdcr_modeling/src"
             "/piecewiseconstantcurvaturemodel.cpp",
             "src/pytdcrsv/tdcr_modeling/src/pseudorigidbodymodel"
             ".cpp",
             "src/pytdcrsv/tdcr_modeling/src"
             "/subsegmentcosseratrodmodel.cpp",
             ],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        include_dirs=extension_headers,
        library_dirs=extension_libraries_dirs,
        libraries=extension_libraries_linking,
        extra_link_args=extra_linking_args,
        extra_compile_args=["-std=c++1z"]
        ),
]

setup(
    name="pytdcrsv",
    version=__version__,
    author="Martin Bensch",
    author_email="Martin.Bensch@imes.uni-hannover.de",
    url="",
    description="",
    long_description="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.6,<2",
        "tqdm",
        "matplotlib",
        "pyquaternion",
        "scipy",
        "pandas",
    ],
    ext_modules=ext_modules,
    extras_require={
        "test": ["pytest"],
        "dev": [
            "jupyterlab",
            "ipympl",
            "ipywidgets",
            "xlsxwriter",
            "openpyxl",
            "pandasgui",
            "numba",
        ],
    },
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9, <3.13",
    include_dirs=extension_headers
)
