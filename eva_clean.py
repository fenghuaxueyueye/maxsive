from image_utils import *
from PIL import Image
from sklearn import metrics
from tqdm import trange
import json
from loader import load_watermark

import json
import torch
import numpy as np
import argparse
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

import torch


def evaluate(image , data , pipe , text_embeddings , watermark , device):
    image_w_distortion = transform_img(image).unsqueeze(0).to(text_embeddings.dtype).to(device)
    image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
    reversed_latents_w = pipe.forward_diffusion(
        latents=image_latents_w,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=50,
    )
    return watermark.detection(reversed_latents_w ,data )

def main(args):
    watermark = load_watermark(args)
    print('eavluating on : ' ,args.watermark_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2-1-base',
            scheduler=scheduler,
            revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)


    positive_path = args.root+ 'positive_image/'
    data_root     = args.root+ 'data/'

    


    positive = []


    for num in trange(args.num):
        num_ = "{:04d}".format(num)
        print('testing on : ',data_root +num_+'.pt')
        data = torch.load(data_root +num_+'.pt' )
        positive_image = Image.open( positive_path + num_+ '.jpg'  )

        positive.append(evaluate(positive_image , data , pipe , text_embeddings , watermark , device))
        # print(positive[-1] , '------')
    

    #tpr metric
    tpr =  str( np.array(positive).mean())
    print('tpr:' ,tpr )
    with open(args.output_path + args.filename, "a") as file:
        file.write( args.watermark_model + ' -  '
            'tpr:' + tpr +    '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--root', default='test/MaXsive/')
    parser.add_argument('--watermark_model', default='MaXsive' , choices=['tree-ring' , 'Gaussian-shading', 'MaXsive' , 'RingID' ])
    parser.add_argument('--num' , default= 100 , type=int)
    parser.add_argument('--tpr_file' , default= None )

    ### watermark args 

    ### gaussian and supershad
    parser.add_argument('--l', default=16, type=int)
    parser.add_argument('--diffusion_bit', default=16, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=2, type=int)
    parser.add_argument('--user_number', default=2, type=int)
    parser.add_argument('--fpr', default= None)

    
    parser.add_argument('--template_c', default=3, type=int)
    parser.add_argument('--distant_func' , default='corr' ,choices=['corr' , 'cos' , 'mse' , 'l1'])
    ### tree-ring
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='ring')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    ### ringid setting
    parser.add_argument('--ring_value_range', default=64, type=int)
    parser.add_argument('--quantization_levels', default=2, type=int)

    parser.add_argument('--output_path' , default= '' , type=str)
    parser.add_argument('--filename' , default= 'clean_result.txt' , type=str)


    args = parser.parse_args()

    main(args)
## python aa_eva_clean.py --root '/work/maopy1998/Gaussian-shading-16bit/' --watermark_model 'Gaussian-shading-general'
## python aa_eva_clean.py --root '/work/maopy1998/RingID/' --watermark_model 'RingID' --tpr_file 'threshold/RingID.pt'
# python aa_eva_clean.py --root '/work/maopy1998/tree-ring-ring/' --watermark_model 'tree-ring' --tpr_file 'threshold/tree-ring-ring.pt'