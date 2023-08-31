# jpeg-decode-and-crop
A custom PyTorch operator that optimizes the jpeg decode and crop flow by decoding only the crop of interest.
The cpp code is based on https://github.com/pytorch/vision/blob/release/0.15/torchvision/csrc/io/image/cpu/decode_jpeg.cpp and the jpeg file is taken from https://github.com/pytorch/vision/blob/96950a5cff2cf3ebee1b91cd3fbafe086d54c9c5/test/assets/encode_jpeg/grace_hopper_517x606.jpg.
## Dependency
The solution requires libjpeg-turbo.
