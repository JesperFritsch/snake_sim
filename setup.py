from setuptools import setup, Extension
import os
import sys

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path.
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the `get_include()`
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

extra_compile_args = []
extra_link_args = []

if os.name == 'nt' and False:
    extra_compile_args.extend(['/Od', '/Zi', '/DDEBUG', '/DPYBIND11_DETAILED_ERROR_MESSAGES'])
    extra_link_args.extend(['/DEBUG'])
else:
    extra_compile_args.extend(['-O0', '-g', '-DDEBUG', '-DPYBIND11_DETAILED_ERROR_MESSAGES'])

if sys.platform == 'win32' or True:
    extra_compile_args.append('/std:c++17')
else:
    extra_compile_args.append('-std=c++11')


ext_modules = [
    Extension(
        'snake_sim.cpp_bindings.area_check',  # Ensure this matches your module path
        [os.path.join('snake_sim', 'cpp_bindings', 'area_check.cpp')],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        language='c++',
        extra_compile_args=['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17'],
    ),
]

setup(
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
)
