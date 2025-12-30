from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vmm_ipc',
    ext_modules=[
        CUDAExtension('vmm_ipc', [
            'ipc_extension.cpp',
        ],
        libraries=['cuda'], # 链接 libcuda.so
        extra_compile_args={'cxx': [], 'nvcc': []})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
