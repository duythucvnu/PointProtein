from setuptools import setup, Extension
import numpy

# Adding OpenCV to project
# ************************

# Adding sources of the project
# *****************************

m_name = "grid_subsampling"

SOURCES = ["cpp_utils/cloud/cloud.cpp", 
           "grid_subsampling/grid_subsampling.cpp",
           "wrapper.cpp"]

module = Extension(m_name,
                   sources=SOURCES,
                   extra_compile_args=['-std=c++11',
                                       '-D_GLIBCXX_USE_CXX11_ABI=0'],
                   include_dirs=[numpy.get_include()])

setup(name=m_name, ext_modules=[module])








