from PIL import Image
import torch
import torchvision.transforms as T
from custom_op.decode_and_crop_jpeg import decode_and_crop_jpeg

h_offset = 10
w_offset = 10
crop_size = 32

img = Image.open('grace_hopper_517x606.jpg')
img = T.PILToTensor()(img)
crop1 = img[:, h_offset:h_offset + crop_size, w_offset:w_offset + crop_size]

with open('grace_hopper_517x606.jpg', 'rb') as f:
    x = torch.frombuffer(f.read(), dtype=torch.uint8)
crop2 = decode_and_crop_jpeg(x, h_offset, w_offset, crop_size, crop_size)

# assertion may fail due to the use of different underlying JPEG libraries
# crops should be compared visually
assert (crop1 == crop2).all()
