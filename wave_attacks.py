from loader import load_dataset
from WAVE_distortions import apply_single_distortion
import torch
import argparse
import os
from tqdm import tqdm
from PIL import Image

attacks_list = ['rotation' , 'resizedcrop' ,'erasing' , 'brightness' , 'contrast' , 'blurring' , 'noise' , 'compression']
attack_strengths = [0.2 , 0.4 ,0.6 , 0.8 , 1.0 ]

def main(args):
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataloader = load_dataset(args)


    data_path           = args.root + 'data/'
    positive_image_path = args.root + 'positive_image/'
    output_image_path   = args.root + 'WAVE_positive_image/'

    os.makedirs(output_image_path, exist_ok=True)


    for i , data in  enumerate(tqdm(dataloader)):
        id_ , prompts = data
        current_prompt = prompts[0]
        num_ = "{:04d}".format(i)
        im = Image.open(positive_image_path+ num_ + '.jpg' )


        for attack in attacks_list:
            for strength in attack_strengths:
                attacked_img = apply_single_distortion(im , attack , strength = strength)

                os.makedirs(output_image_path + num_  +'/', exist_ok=True)
                attacked_img.save(output_image_path + num_  +'/'  +num_ + '_' + attack + '_' + str(strength).replace('.' , '_') + '.jpg')




# CUDA_VISIBLE_DEVICES=1 python generate_SuperShad_for_str_mark.py --output_path './SuperShad_output_strmark/round2/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    ### output config
    parser.add_argument('--root', default='./output/')
    ### dataset
    parser.add_argument('--dataset', default='ImageNet')
    parser.add_argument('--num_of_imgs', default=1, type=int)
    parser.add_argument('--start_num', default=0, type=int)
    parser.add_argument('--end_num', default=999, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--shuffle' , default = False)
    

    args = parser.parse_args()
    main(args)
# python a_image_generate.py --watermark_model 'RingId' --output_path ''
# a_wave_attacks.py --root 'SuperShad_mean_12_5'