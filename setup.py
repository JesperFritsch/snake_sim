from setuptools import setup, Extension
from setuptools.command.build_py import build_py
import os
import sys
import configparser
from grpc_tools import protoc
import glob

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

class build_proto(build_py):
    user_options = build_py.user_options + [
        ('output_path=', None, 'output path for generated python files'),
        ('proto_files=', None, 'proto files to compile')
    ]

    def initialize_options(self):
        build_py.initialize_options(self)
        self.output_path = None
        self.proto_files = None
        
    def finalize_options(self):
        build_py.finalize_options(self)
        config = configparser.ConfigParser()
        config.read('setup.cfg')
        if self.output_path is None:
            self.output_path = config['build_proto']['output_path']
        if self.proto_files is None:
            self.proto_files = config['build_proto']['proto_files'].split()
        
    def run(self):
        self.run_command("build_ext")
        print(f'Compiling proto files: {self.proto_files}')
        # check if the self.proto_files is a glob pattern
        proto_files = []
        for proto_file in self.proto_files:
            if '*' in proto_file:
                proto_files.extend(glob.glob(proto_file))
            else:
                proto_files.append(proto_file)
                
        for proto_file in proto_files:
            protoc.main([
                'grpc_tools.protoc',
                f'--python_out={self.output_path}',
                f'--proto_path=.',
                proto_file
            ])
        build_py.run(self)

extra_compile_args = []
extra_link_args = []

if os.name == 'nt':
    extra_compile_args.extend(['/Od', '/Zi', '/DDEBUG', '/DPYBIND11_DETAILED_ERROR_MESSAGES'])
    extra_link_args.extend(['/DEBUG'])
else:
    extra_compile_args.extend(['-O0', '-g', '-DDEBUG', '-DPYBIND11_DETAILED_ERROR_MESSAGES'])

if sys.platform == 'win32':
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
    cmdclass={'build_py': build_proto},
)
