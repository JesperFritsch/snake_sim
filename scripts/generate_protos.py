#!/usr/bin/env python3
"""
Script to generate Python protobuf modules from .proto files using grpc_tools.protoc.
Reads proto_files and output_path from [tool.build_proto] in pyproject.toml.
"""
import os
import sys
import glob
import tomllib
from grpc_tools import protoc

def main():
    # Find pyproject.toml in the project root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject_path = os.path.join(root, 'pyproject.toml')
    with open(pyproject_path, 'rb') as f:
        config = tomllib.load(f)
    build_proto_config = config['tool']['build_proto']
    proto_files = build_proto_config['proto_files']
    output_path = build_proto_config['output_path']

    # Expand globs
    all_proto_files = []
    for proto_file in proto_files:
        if '*' in proto_file:
            all_proto_files.extend(glob.glob(proto_file))
        else:
            all_proto_files.append(proto_file)

    if not all_proto_files:
        print('No .proto files found to compile.')
        sys.exit(1)

    for proto_file in all_proto_files:
        print(f"Compiling {proto_file} -> {output_path}")
        result = protoc.main([
            'grpc_tools.protoc',
            f'--python_out={output_path}',
            f'--proto_path=.',
            proto_file
        ])
        if result != 0:
            print(f"Failed to compile {proto_file}")
            sys.exit(result)
    print("All proto files compiled successfully.")

if __name__ == "__main__":
    main()
