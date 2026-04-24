# import torch
# import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import pad
# from PIL import Image, ImageFilter
from WAVE_distortions import apply_single_distortion




def transform_img(image, target_size=512):
#         print('------' , image.size)
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


# def latents_to_imgs(pipe, latents):
#     x = pipe.decode_image(latents)
#     x = pipe.torch_to_numpy(x)
#     x = pipe.numpy_to_pil(x)
#     return x


def image_distortion(img, args):
    if args.WAVE_rotation is not None:
        img = apply_single_distortion(img , distortion_type= 'rotation' , strength = args.WAVE_rotation)

    if args.WAVE_resizedcrop is not None:
        # print('rooooooooooootation')
        img = apply_single_distortion(img , distortion_type= 'resizedcrop' , strength = args.WAVE_resizedcrop)
   
    if args.WAVE_erasing is not None:
        img = apply_single_distortion(img , distortion_type= 'erasing' , strength = args.WAVE_erasing)

    if args.WAVE_brightness is not None:
        img = apply_single_distortion(img , distortion_type= 'brightness' , strength = args.WAVE_brightness)
    
    if args.WAVE_contrast is not None:
        img = apply_single_distortion(img , distortion_type= 'contrast' , strength = args.WAVE_contrast)
    if args.WAVE_blurring is not None:
        img = apply_single_distortion(img , distortion_type= 'blurring' , strength = args.WAVE_blurring)

    if args.WAVE_noise is not None:
        img = apply_single_distortion(img , distortion_type= 'noise' , strength = args.WAVE_noise)

    if args.WAVE_compression is not None:
        img = apply_single_distortion(img , distortion_type= 'compression' , strength = args.WAVE_compression)
    
    # if args.flip is not None:
    #     print('Image fliping')
    #     img = apply_single_distortion(img , distortion_type= 'flip' , strength = 0)
    
    # if args.sharpness is not None:
    #     print('Image sharpness')
    #     img = apply_single_distortion(img , distortion_type= 'sharpness' , strength = 0)

    return img

