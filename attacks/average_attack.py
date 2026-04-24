from dataclasses import dataclass
# from typing import List, Callable, Dict, Any
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy


@dataclass
class Pattern:
    num_images: int
    pattern: np.ndarray
class average_attack:
    def __init__(self , pth_w_img , pth_no_w_img , average_list = [10, 50 , 100 ,1000]):
        self.pth_w_img = pth_w_img
        self.pth_no_w_img = pth_no_w_img

        self.w_img_list = self.sum_images(self.pth_w_img , average_list , (512 ,512))
        self.no_w_img_list = self.sum_images(self.pth_no_w_img , average_list , (512 ,512))
        self.blackbox_watermarks = self.get_difference_list(
                self.no_w_img_list,
                self.w_img_list,
                average_list
            )
        self.average_list = average_list

    def sum_images(self, image_path, num_images, image_size):
        '''Returns a list of avged images [5 img avg, 10 img avg, 20 img avg, etc.]'''

        image_sum = None
        image_sum_list = []
        count_image = 0

        files = os.listdir(image_path)
        files = sorted(files)#[:max(num_images)]

        for file in tqdm(files):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image = np.array(Image.open(os.path.join(image_path, file)).resize(image_size))
                if len(image.shape)==3:
                    if image_sum is None:
                        image_sum = image.astype(float)
                    else:
                        image_sum += image.astype(float)
                    count_image += 1

                    if count_image in num_images:
                        image_sum_list.append(Pattern(count_image, image_sum / count_image))
            if len(image_sum_list) == len(num_images):
                return image_sum_list
    def get_difference_list(self , clean_image_list, watermark_image_list, num_images):
        difference_list = []
        for num_images_index in range(len(num_images)):
            assert clean_image_list[num_images_index].num_images == watermark_image_list[num_images_index].num_images
            difference_list.append(Pattern(
                num_images[num_images_index],
                watermark_image_list[num_images_index].pattern - clean_image_list[num_images_index].pattern
            ))
        return difference_list
        
    def scale_removal(self , image: np.ndarray, watermark: np.ndarray, factor: int, watermark_bound: int, sign: bool = False, random_flip: bool = False):
        image = copy.deepcopy(image)
        watermark = copy.deepcopy(watermark)
        if sign:
            watermark = np.sign(watermark)
        if random_flip:
            watermark *= np.sign(np.random.randn(*watermark.shape))
        if watermark_bound is not None:
            return np.clip(image - np.clip(factor * watermark, -watermark_bound, watermark_bound), 0, 255)
        return np.clip(image - factor * watermark, 0, 255)
    def removal(self, img , num_average):
        img = self.scale_removal(np.array(img) , self.blackbox_watermarks[self.average_list.index(num_average)].pattern , 1 , None )
        return Image.fromarray(np.uint8(img))
    def forgery(self, img , num_average):
        img = self.scale_removal(np.array(img) , self.blackbox_watermarks[self.average_list.index(num_average)].pattern , -1 , None )
        return Image.fromarray(np.uint8(img))