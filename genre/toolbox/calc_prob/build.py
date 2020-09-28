import os
import sys
import torch
from torch.utils.ffi import create_extension


this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

extra_compile_args = list()


extra_objects = list()
assert(torch.cuda.is_available())
sources = ['calc_prob/src/calc_prob.c']
headers = ['calc_prob/src/calc_prob.h']
defines = [('WITH_CUDA', True)]
with_cuda = True

extra_objects = ['calc_prob/src/calc_prob_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi_params = {
    'headers': headers,
    'sources': sources,
    'define_macros': defines,
    'relative_to': __file__,
    'with_cuda': with_cuda,
    'extra_objects': extra_objects,
    'include_dirs': [os.path.join(this_file, 'calc_prob/src')],
    'extra_compile_args': extra_compile_args,
}


if __name__ == '__main__':
    ext = create_extension(
        'calc_prob._ext.calc_prob_lib',
        package=False,
        **ffi_params)
    #from setuptools import setup
    # setup()
    ext.build()

    # ffi.build()
