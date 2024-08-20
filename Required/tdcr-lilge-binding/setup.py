import sys
from setuptools import setup, find_packages
from sys import platform
import os

from pybind11 import get_cmake_dir
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext


print(" ############# ")
print(" \n \n Does it work \n \n")
if platform == "linux" or platform == "linux2":
    # linux
    extension_headers = ['/usr/include/eigen3',
                         '/usr/include/eigen',
                         '/usr/include/',
                         '/usr/opt/',
                         '/usr/opt/eigen',
                         '/usr/opt/eigen3',
                         '/src/tdcr-modeling/include/',
                         '/src/eigen/'
                         ]

    extension_libraries_dirs = ["/usr/lib/", "/usr/gsl-2.7/gsl_27/lib","/usr/gsl-2.7/.libs/"]
    extension_runtime_libraries_dir = ["/usr/lib/", "/usr/gsl-2.7/gsl_27/lib", "/usr/gsl-2.7/.libs/"]
    extension_libraries_linking = ["gsl", "gslcblas"]
    extra_linking_args = ["-Wl,--no-as-needed"]

    try:
        os.system("echo $LD_LIBRARY_PATH")
        os.system("ldconfig -v")
    except:
        os.system("echo error in setup.py")

    print("System is Linux!")
elif platform == "darwin":
    # OS X
    extension_headers = ['/usr/local/include/eigen3',
                         '/usr/local/include/',
                         '/src/tdcr-modeling/include/'
                         ]

    extension_libraries_dirs = ["/usr/local/lib/", "/usr/local/Cellar/gsl/2.7.1/lib/"]
    extension_runtime_libraries_dirs = ["/usr/local/lib/", "/usr/local/Cellar/gsl/2.7.1/lib/"]
    extension_libraries_linking = ["gsl", "gslcblas"]
    extra_linking_args = []#["-L/usr/local/lib -lgsl -lgslcblas"]
    print("System is darwin!")

elif platform == "win32":
    raise NotImplementedError("Windows is currently not supported.")
else:
    print("On system: " + platform)
print(" ############# ")

__version__ = "1.10.0"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)



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
        #runtime_library_dirs=extension_runtime_libraries_dirs,
        library_dirs=extension_libraries_dirs,
        libraries=extension_libraries_linking,
       extra_link_args=extra_linking_args,#"-L/usr/local/lib", "-lgsl",
                      # "-lgslcblas"],
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
                            #1.18"
    install_requires=["py", "numpy==1.21.6", "tqdm", "matplotlib",
                      "pyquaternion", "scipy",
                      "pandas", "numba", "xlsxwriter", "openpyxl",
                      "jupyterlab", "ipympl", "ipywidgets", "pandasgui"],
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7, <3.11",
    #for mac, since Eigen is located under eigen3/....
    include_dirs=extension_headers
)
