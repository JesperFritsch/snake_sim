from setuptools import setup, Extension
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext
import os
import sys
import tomllib
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

class build_ext_parallel(build_ext):
    def finalize_options(self):
        super().finalize_options()
        # Use all available cores unless the user already specified -j/--parallel.
        if not self.parallel:
            self.parallel = os.cpu_count() or 1

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
        with open('pyproject.toml', 'rb') as f:
            config = tomllib.load(f)
        build_proto_config = config['tool']['build_proto']
        if self.output_path is None:
            self.output_path = build_proto_config['output_path']
        if self.proto_files is None:
            self.proto_files = build_proto_config['proto_files']

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

debug_build = os.environ.get("DEBUG", "0") == "1"

if debug_build:
    if os.name == 'nt':
        extra_compile_args.extend(['/Od', '/Zi', '/DDEBUG', '/DPYBIND11_DETAILED_ERROR_MESSAGES'])
        extra_link_args.extend(['/DEBUG'])
    else:
        extra_compile_args.extend(['-O0', '-g', '-DDEBUG', '-DPYBIND11_DETAILED_ERROR_MESSAGES'])
else:
    if os.name == 'nt':
        extra_compile_args.extend(['/Ox'])
    else:
        extra_compile_args.extend(['-O3', '-march=native', '-flto'])

if sys.platform == 'win32':
    extra_compile_args.append('/std:c++17')
else:
    extra_compile_args.append('-std=c++17')


ext_modules = [
    Extension(
        'snake_sim.cpp_bindings.area_check',
        # area_check needs its own sources plus the utils *implementation* files,
        # but not pybind_utils.cpp (that defines the separate `utils` module and
        # would bake a stray PyInit_utils into this extension).
        sorted(
            f for f in glob.glob('snake_sim/cpp_bindings/**/src/*.cpp')
            if os.path.basename(f) != 'pybind_utils.cpp'
        ),
        include_dirs=[
            'snake_sim/cpp_bindings/area_check/include',
            'snake_sim/cpp_bindings/utils/include',
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        'snake_sim.cpp_bindings.utils',
        # Use glob to automatically include all .cpp files in the src directorye
        sorted(glob.glob('snake_sim/cpp_bindings/utils/src/*.cpp')),
        include_dirs=[
            'snake_sim/cpp_bindings/utils/include',
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_py': build_proto, 'build_ext': build_ext_parallel},
)
