import os
import sys
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

extra_compile_args = list()

extra_objects = list()
assert(torch.cuda.is_available())
sources = ['cam_bp/src/back_projection.c']
headers = ['cam_bp/src/back_projection.h']
defines = [('WITH_CUDA', True)]
with_cuda = True

extra_objects = ['cam_bp/src/back_projection_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi_params = {
    'headers': headers,
    'sources': sources,
    'define_macros': defines,
    'relative_to': __file__,
    'with_cuda': with_cuda,
    'extra_objects': extra_objects,
    'include_dirs': [os.path.join(this_file, 'cam_bp/src')],
    'extra_compile_args': extra_compile_args,
}

ffi = create_extension(
    'cam_bp._ext.cam_bp_lib',
    package=True,
    **ffi_params
)

if __name__ == '__main__':
    ffi = create_extension(
        'cam_bp._ext.cam_bp_lib',
        package=False,
        **ffi_params)
    ffi.build()
