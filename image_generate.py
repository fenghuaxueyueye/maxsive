from loader import load_gen_model, load_watermark,load_dataset
import torch
import argparse
import os
from tqdm import tqdm

def main(args):
    # load diffusion model
    pipe = load_gen_model(args.model_id)
    watermark_models = load_watermark(args)
    dataloader = load_dataset(args)


    data_path           = args.output_path + 'data/'
    positive_image_path = args.output_path + 'positive_image/'

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(positive_image_path, exist_ok=True)



    for i , data in  enumerate(tqdm(dataloader)):
        id_ , prompts = data
        current_prompt = prompts[0]

        # make watermark initial noise z and watermark data
        z , data = watermark_models.watermark_injection()
        
        outputs = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            latents=z
        )
        positive_image = outputs.images[0]


        num_ = "{:04d}".format(i)
        print('saving img at :' , positive_image_path + num_ + '.jpg')
        print('saving watermark data at :' , data , data_path + num_ + '.pt')
        positive_image.save(positive_image_path + num_ + '.jpg')
        torch.save(data , data_path + num_ + '.pt')





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    ### watermark setting 
    parser.add_argument('--diffusion_bit', default=16, type=int)
    parser.add_argument('--watermark_model', default='RingID' , choices=['tree-ring' , 'Gaussian-shading', 'MaXsive' , 'RingID' ])
    ### gaussian shading
    parser.add_argument('--l', default=16, type=int)
    parser.add_argument('--template_c', default=3, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=2, type=int)
    parser.add_argument('--user_number', default=2, type=int)
    parser.add_argument('--fpr', default=0.001)
    parser.add_argument('--distant_func' , default='corr' ,choices=['corr' , 'cos' , 'mse' , 'l1'])
    ### ringid setting
    parser.add_argument('--ring_value_range', default=64, type=int)
    parser.add_argument('--quantization_levels', default=2, type=int)
    parser.add_argument('--tpr_file', default=None)
    ### tree-ring setting
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='all')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    ### output config
    parser.add_argument('--output_path', default='./output/')

    parser.add_argument('--model_id', default='SD21')
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)    
    ### dataset
    parser.add_argument('--shuffle', default=False, type=bool)
    parser.add_argument('--dataset', default='ImageNet')
    parser.add_argument('--num_of_imgs', default=1, type=int)
    parser.add_argument('--start_num', default=0, type=int)
    parser.add_argument('--end_num', default=99, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    

    args = parser.parse_args()
    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps
    main(args)
# python aa_image_generate.py --watermark_model 'tree-ring' --output_path '/work/maopy1998/tree-ring-ring/' --w_pattern 'ring' --w_mask_shape 'circle'

# python aa_image_generate.py --watermark_model 'Super-shading' --output_path '/work/maopy1998/supershad_1_8_t7/' --num_of_imgs 10 --hw_copy 8 --model_id 'watermarkSD21'
