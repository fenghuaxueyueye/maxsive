from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# Normalize image tensors
def normalize_tensor(images, norm_type):
    assert norm_type in ["imagenet", "naive"]
    # Two possible normalization conventions
    if norm_type == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean, std)
    elif norm_type == "naive":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        normalize = transforms.Normalize(mean, std)
    else:
        assert False
    return torch.stack([normalize(image) for image in images])


# Unnormalize image tensors
def unnormalize_tensor(images, norm_type):
    assert norm_type in ["imagenet", "naive"]
    # Two possible normalization conventions
    if norm_type == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        unnormalize = transforms.Normalize(
            (-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
            (1 / std[0], 1 / std[1], 1 / std[2]),
        )
    elif norm_type == "naive":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        unnormalize = transforms.Normalize(
            (-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
            (1 / std[0], 1 / std[1], 1 / std[2]),
        )
    else:
        assert False
    return torch.stack([unnormalize(image) for image in images])
def to_tensor(images, norm_type=None):
    assert isinstance(images, list) and all(
        [isinstance(image, Image.Image) for image in images]
    )
    images = torch.stack([transforms.ToTensor()(image) for image in images])
    if norm_type is not None:
        images = normalize_tensor(images, norm_type)
    return images


# Unnormalize tensors and convert to PIL images
def to_pil(images, norm_type="naive"):
    assert isinstance(images, torch.Tensor)
    if norm_type is not None:
        images = unnormalize_tensor(images, norm_type).clamp(0, 1)
    return [transforms.ToPILImage()(image) for image in images.cpu()]



def decode(latents , pipe):
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[
                0
            ]
    image = pipe.image_processor.postprocess(image.cpu().detach(), output_type='pil')
    return image

