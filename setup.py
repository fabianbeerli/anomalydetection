from setuptools import setup, Extension
import sys
import os
import pybind11

# Ensure pybind11 is imported correctly
pybind11_include_dir = pybind11.get_include()

# Source files for the extension
sources = [
    'C++/aida_wrapper.cpp',
    'C++/src/aida_class.cpp',
    'C++/src/distance_metrics.cpp',
    'C++/src/isolation_formulas.cpp',
    'C++/src/aggregation_functions.cpp',
    'C++/src/rng_class.cpp'
]

# C++ extension module
ext_modules = [
    Extension(
        'aida_cpp',
        sources=sources,
        include_dirs=[
            # Path to pybind11 headers
            pybind11_include_dir,
            pybind11.get_include(user=True),
            'C++/include'
        ],
        language='c++',
        extra_compile_args=['-std=c++11', '-O3']
    ),
]

setup(
    name='aida_cpp',
    version='0.1.0',
    author='Fabian Beerli',
    description='Python bindings for AIDA C++ implementation',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    install_requires=['pybind11>=2.5.0'],
    zip_safe=False,
)