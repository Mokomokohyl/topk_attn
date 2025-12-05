from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='retrieval_attention',
    ext_modules=[
        CUDAExtension(
            name='retrieval_attention_cpp',
            sources=['bind.cpp', 'retrieval_attention.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_120a',
                    '--ptxas-options=-v',
                    '-lineinfo'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
