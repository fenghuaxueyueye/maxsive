from loader import load_gen_model, load_dataset
import torch
import argparse
import os
from tqdm import tqdm

def main(args):
    # load diffusion model
    pipe = load_gen_model(args.model_id)
    dataloader = load_dataset(args)


    data_path           = args.output_path + 'data/'
    image_path = args.output_path + 'images/'

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)



    for i , data in  enumerate(tqdm(dataloader)):
        id_ , prompts = data
        current_prompt = prompts[0]
        z = torch.randn(1,4,64,64).half()
        
        outputs = pipe(
            current_prompt,
            latents=z
        )
        positive_image = outputs.images[0]


        num_ = "{:04d}".format(i)

        positive_image.save(image_path + num_ + '.jpg')
        torch.save(data , data_path + num_ + '.pt')




# CUDA_VISIBLE_DEVICES=1 python generate_SuperShad_for_str_mark.py --output_path './SuperShad_output_strmark/round2/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    ### output config
    parser.add_argument('--output_path', default='./output/')

    parser.add_argument('--model_id', default='SD21')
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)    
    ### dataset
    parser.add_argument('--dataset', default='ImageNet')
    parser.add_argument('--num_of_imgs', default=1, type=int)
    parser.add_argument('--start_num', default=0, type=int)
    parser.add_argument('--end_num', default=999, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--shuffle', default=False, type=bool)
    

    args = parser.parse_args()
    main(args)
# python a_image_generate.py --watermark_model 'RingId' --output_path ''
