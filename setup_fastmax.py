from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fastmax_cuda',
    ext_modules=[
        CUDAExtension('fastmax_cuda', [
            'fastmax_cuda.cpp',
            'fastmax_cuda_forward.cu',
            'fastmax_cuda_backward.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
