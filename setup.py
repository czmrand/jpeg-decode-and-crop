from setuptools import setup
from torch.utils import cpp_extension

setup(name='decode_and_crop_jpeg',
      ext_modules=[cpp_extension.CppExtension('decode_and_crop_jpeg', 
                                              ['decode_and_crop_jpeg.cpp'], 
                                              libraries=['jpeg'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
