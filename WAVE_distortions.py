## copy from WAVES
import random
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import torch
import io
from utils import to_tensor, to_pil

distortion_strength_paras = dict(
    rotation=(0, 45),
    resizedcrop=(1, 0.5),
    erasing=(0, 0.25),
    brightness=(1, 2),
    contrast=(1, 2),
    blurring=(0, 20),
    noise=(0, 0.1),
    compression=(90, 10),
    flip = (0,1),
    sharpness = (0,2),
    gaussaan_blur = (0,3)
)

def relative_strength_to_absolute(strength, distortion_type):
    assert 0 <= strength <= 1
    strength = (
        strength
        * (
            distortion_strength_paras[distortion_type][1]
            - distortion_strength_paras[distortion_type][0]
        )
        + distortion_strength_paras[distortion_type][0]
    )
    strength = max(strength, min(*distortion_strength_paras[distortion_type]))
    strength = min(strength, max(*distortion_strength_paras[distortion_type]))
    return strength

def apply_single_distortion(image, distortion_type, strength=None, distortion_seed=0,relative_strength=True):
    # Accept a single image
    assert isinstance(image, Image.Image)
    # Set the random seed for the distortion if given
#     set_random_seed(distortion_seed)
    # Assert distortion type is valid
    assert distortion_type in distortion_strength_paras.keys()
    # Assert strength is in the correct range
#     if strength is not None:
#         assert (
#             min(*distortion_strength_paras[distortion_type])
#             <= strength
#             <= max(*distortion_strength_paras[distortion_type])
#         )
    if relative_strength:
        strength = relative_strength_to_absolute(strength, distortion_type)
    # Apply the distortion
    if distortion_type == "rotation":
        angle = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["rotation"])
        )
        distorted_image = F.rotate(image, angle)

    elif distortion_type == "resizedcrop":
        scale = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["resizedcrop"])
        )
        i, j, h, w = T.RandomResizedCrop.get_params(
            image, scale=(scale, scale), ratio=(1, 1)
        )
        distorted_image = F.resized_crop(image, i, j, h, w, image.size)

    elif distortion_type == "erasing":
        scale = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["erasing"])
        )
        image = to_tensor([image], norm_type=None)
        i, j, h, w, v = T.RandomErasing.get_params(
            image, scale=(scale, scale), ratio=(1, 1), value=[0]
        )
        distorted_image = F.erase(image, i, j, h, w, v)
        distorted_image = to_pil(distorted_image, norm_type=None)[0]

    elif distortion_type == "brightness":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["brightness"])
        )
        enhancer = ImageEnhance.Brightness(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "contrast":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["contrast"])
        )
        enhancer = ImageEnhance.Contrast(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "blurring":
        kernel_size = (
            int(strength)
            if strength is not None
            else random.uniform(*distortion_strength_paras["blurring"])
        )
        distorted_image = image.filter(ImageFilter.GaussianBlur(kernel_size))

    elif distortion_type == "noise":
        std = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["noise"])
        )
        image = to_tensor([image], norm_type=None)
        noise = torch.randn(image.size()) * std
        distorted_image = to_pil((image + noise).clamp(0, 1), norm_type=None)[0]

    elif distortion_type == "compression":
        quality = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["compression"])
        )
        quality = int(quality)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        distorted_image = Image.open(buffered)
        
    elif distortion_type == "flip":
        image = to_tensor([image], norm_type=None) ## 1 , 3 ,512  ,512
#         print(image.shape)
        distorted_image = torch.flip(image , [3])
        distorted_image = to_pil(distorted_image.clamp(0, 1), norm_type=None)[0]
    elif distortion_type == "sharpness":
        image = to_tensor([image], norm_type=None)
        distorted_image = T.functional.adjust_sharpness(image,2)
        distorted_image = to_pil(distorted_image.clamp(0, 1), norm_type=None)[0]
    
    elif distortion_type == "gaussaan_blur":
        image = to_tensor([image], norm_type=None)
        distorted_image = T.functional.gaussian_blur(image,3)
        distorted_image = to_pil(distorted_image.clamp(0, 1), norm_type=None)[0]
        
    else:
        assert False

    return distorted_image