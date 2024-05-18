import os
import torch
from torch.utils.cpp_extension import load

EXTENSION_DIR = os.path.join(os.path.dirname(__file__))

_exts = {}

def _load_or_compile(num_neighbors, rebuild=False):
    
    extension_name = f'knn_{num_neighbors}'
    build_dir = os.path.join(EXTENSION_DIR, "build")
    if not os.path.exists(build_dir): os.mkdir(build_dir)
    
    src_path = os.path.join(EXTENSION_DIR, "csrc")
    extension_path = os.path.join(src_path, f"{extension_name}.cu")

    if not os.path.exists(extension_path) or rebuild:
        with open(os.path.join(src_path, "simple_knn.cu.template")) as f:
            extension_code = f.read()
        extension_code = extension_code.replace("{{num_nearest}}", str(num_neighbors))
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
        verbose=rebuild
    )

def knn(points, num_knn, rebuild=False):
    ext = _exts.get(num_knn)
    if ext is None or rebuild:
        ext = _exts[num_knn] = _load_or_compile(num_knn, rebuild=rebuild)
        
    return ext.distCUDA2(points)