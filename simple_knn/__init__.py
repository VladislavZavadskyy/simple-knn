import os
import torch
from torch.utils.cpp_extension import load

__all__ = ["knn"]

EXTENSION_DIR = os.path.join(os.path.dirname(__file__))

_exts = {}

def _load_or_compile(num_neighbors, rebuild=False, verbose=False):
    
    extension_name = f'knn_{num_neighbors}'
    build_dir = os.path.join(EXTENSION_DIR, "build")
    if not os.path.exists(build_dir): os.mkdir(build_dir)
    
    src_path = os.path.join(EXTENSION_DIR, "csrc")
    extension_path = os.path.join(src_path, f"{extension_name}.cu")

    with open(os.path.join(src_path, "simple_knn.cu.template")) as f:
        extension_code = f.read()
    extension_code = extension_code.replace("{{num_nearest}}", str(num_neighbors))
    
    code_does_not_match = True
    if os.path.exists(extension_path):
        with open(extension_path) as f:
            code_does_not_match = f.read() != extension_code
    
    if code_does_not_match or rebuild:
        with open(extension_path, 'w') as f:
            f.write(extension_code)

    cxx_compiler_flags = ['-O3']
    if os.name == 'nt':
        cxx_compiler_flags.append("/wd4624")
        
    return load(
        name=extension_name,
        sources=[extension_path],
        extra_cflags=cxx_compiler_flags,
        extra_cuda_cflags=["-O3"],
        build_directory=build_dir,
        is_python_module=True,
        verbose=verbose
    )

def knn(points, num_knn):
    ext = _exts.get(num_knn)
    if ext is None:
        ext = _exts[num_knn] = _load_or_compile(num_knn)
        
    return ext.distCUDA2(points)