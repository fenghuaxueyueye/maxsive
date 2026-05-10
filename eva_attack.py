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


def evaluate(image , data , pipe , text_embeddings , watermark , device, eval_bit_acc=False):
    image_w_distortion = transform_img(image).unsqueeze(0).to(text_embeddings.dtype).to(device)
    image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
    reversed_latents_w = pipe.forward_diffusion(
        latents=image_latents_w,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=50,
    )
    result = {
        'tpr': watermark.detection(reversed_latents_w ,data )
    }
    if eval_bit_acc:
        result.update(watermark.bit_accuracy(reversed_latents_w, data))
    return result

def main(args):
    watermark = load_watermark(args)
    print('eavluating on : ' ,args.watermark_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2-1-base',
            scheduler=scheduler,
            torch_dtype=args.diffusion_bit,
            revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)


    #### load work list 
    # Open and read the JSON file
    with open(args.distortion_list, 'r') as file:
        print('working on ' , args.distortion_list)
        distortion_type = json.load(file)


    positive_path = args.root+ args.attacked_path
    data_root     = args.root+ 'data/'

    


    for distortion in distortion_type:
        print( 'working on ', distortion)
        tpr = []
        bit_acc = []
        ber = []

        for num in trange(args.num):
            num_ = "{:04d}".format(num)
            print('evaluating on : ', data_root +num_+'.pt' , 'image : '  , positive_path + num_+ '/'+ num_+ '_'+ distortion_type[distortion]  )
            data = torch.load(data_root +num_+'.pt' )
            positive_image = Image.open( positive_path + num_+ '/'+ num_+ '_'+ distortion_type[distortion]  )

            eval_result = evaluate(positive_image , data , pipe , text_embeddings , watermark , device, args.eval_bit_acc)
            tpr.append(eval_result['tpr'])
            if args.eval_bit_acc:
                bit_acc.append(eval_result['bit_acc'])
                ber.append(eval_result['ber'])
                print( 'tpr : ' , np.array(tpr).mean()  , ' tpr last : ' , tpr[-1], ' bit_acc : ', np.array(bit_acc).mean(), ' bit_acc last : ', bit_acc[-1], ' ber : ', np.array(ber).mean(), ' ber last : ', ber[-1])
            else:
                print( 'tpr : ' , np.array(tpr).mean()  , ' tpr last : ' , tpr[-1])
        

        output_line = str(distortion) + ' -  ' + 'tpr:' + str(np.array(tpr).mean())
        if args.eval_bit_acc:
            output_line += ' bit_acc:' + str(np.array(bit_acc).mean()) + ' ber:' + str(np.array(ber).mean())

        with open(args.output_path + args.filename, "a") as file:
            file.write(output_line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--root', default='/work/maopy/SuperShadeC3/')
    parser.add_argument('--attacked_path', default='regen_positive_image/')
    parser.add_argument('--watermark_model', default='MaXsive' , choices=['tree-ring' , 'Gaussian-shading', 'MaXsive' , 'RingID' ])
    parser.add_argument('--num' , default= 100 , type=int)
    parser.add_argument('--tpr_file' , default= None )
    parser.add_argument('--eval_bit_acc', action='store_true', help='evaluate bit accuracy and bit error rate for MaXsive watermark')

    parser.add_argument('--diffusion_bit', default=16, type=int)
    ### watermark args 

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
    ### tree-ring setting
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='ring')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    ### output config
    parser.add_argument('--output_path', default='')
    parser.add_argument('--filename', default='MaXsive_WAVEs_result.txt')
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
    
    parser.add_argument('--distortion_list'  , type=str)

    args = parser.parse_args()
    if args.diffusion_bit == 32:
        args.diffusion_bit = torch.float32
    if args.diffusion_bit == 16:
        args.diffusion_bit = torch.float16
    main(args)


 # python aa_eva_attack.py --distortion_list 'work0_stirmark_distort_type.json' --root '/work/maopy1998/tree-ring-stirmark/' --attacked_path 'pos_stirmark_image/'  --watermark_model 'tree-ring' --filename 'tree-ring-stirmark.txt' --output_path 'last_output/' --tpr_file 'threshold/tree-ring.pt'
    
#python aa_eva_attack.py --distortion_list 'WAVEs_p3.json' --root '/work/maopy1998/Gaussain_optimal_verification/' --attacked_path 'WAVE_positive_image/'  --watermark_model 'Gaussian-shading' --filename 'Gaussian-shad-origin-WAVEs.txt' --output_path 'final_result/'
# python aa_eva_attack.py --distortion_list 'work0_stirmark_distort_type.json' --root '/work/maopy1998/tree-ring-stirmark/' --attacked_path 'pos_stirmark_image/'  --watermark_model 'tree-ring' --filename 'tree-ring-stirmark.txt' --output_path 'last_output/' --tpr_file 'threshold/tree-ring.pt'
 # python aa_eva_attack.py --distortion_list 'work1_stirmark_distort_type.json' --root '/work/maopy1998/tree-ring-stirmark/' --attacked_path 'pos_stirmark_image/'  --watermark_model 'tree-ring' --filename 'tree-ring-stirmark.txt' --output_path 'last_output/' --tpr_file 'threshold/tree-ring.pt'
 # python aa_eva_attack.py --distortion_list 'work2_stirmark_distort_type.json' --root '/work/maopy1998/tree-ring-stirmark/' --attacked_path 'pos_stirmark_image/'  --watermark_model 'tree-ring' --filename 'tree-ring-stirmark.txt' --output_path 'last_output/' --tpr_file 'threshold/tree-ring.pt'
 # python aa_eva_attack.py --distortion_list 'work3_stirmark_distort_type.json' --root '/work/maopy1998/tree-ring-stirmark/' --attacked_path 'pos_stirmark_image/'  --watermark_model 'tree-ring' --filename 'tree-ring-stirmark.txt' --output_path 'last_output/' --tpr_file 'threshold/tree-ring.pt'
# python aa_eva_attack.py --distortion_list 'stirmark_RST.json' --root '/work/maopy1998/SuperShad_mean_12_5/' --attacked_path 'pos_stirmark_image/'  --watermark_model 'Super-shading' --filename 'Super-shading-RST-stirmark.txt' --output_path 'last_output/' --hw_copy 2 --tpr_file 'threshold/super-shad.pt'
##identify tree-ring
# python aa_eva_attack.py --distortion_list 'WAVEs_adv_use.json' --root '/work/maopy1998/tree-ring-stirmark/' --attacked_path 'adv_positive_image/'  --watermark_model 'tree-ring' --filename 'tree-ring-adv.txt' --output_path 'last_output/' --tpr_file 'threshold/tree-ring.pt'
#  python aa_eva_attack.py --distortion_list 'WAVEs_regen_use.json' --root '/work/maopy1998/tree-ring-stirmark/' --attacked_path 'regen_positive_image/'  --watermark_model 'tree-ring' --filename 'tree-ring-adv.txt' --output_path 'last_output/' --tpr_file 'threshold/tree-ring.pt'
# python aa_eva_attack.py --distortion_list 'WAVEs_adv_use.json' --root '/work/maopy1998/tree-ring-16bit/' --attacked_path 'adv_positive_image/'  --watermark_model 'tree-ring' --filename 'tree-ring-high-adv.txt' --output_path 'last_output/' --tpr_file 'threshold/tree-ring.pt'
#  python aa_eva_attack.py --distortion_list 'WAVEs_regen_use.json' --root '/work/maopy1998/tree-ring-16bit/' --attacked_path 'regen_positive_image/'  --watermark_model 'tree-ring' --filename 'tree-ring-high-adv.txt' --output_path 'last_output/' --tpr_file 'threshold/tree-ring.pt'
### ringid

# python aa_eva_attack.py --distortion_list 'WAVEs_adv_use.json' --root '/work/maopy1998/RingID/' --attacked_path 'adv_positive_image/'  --watermark_model 'RingID' --filename 'RingID-adv.txt' --output_path 'last_output/' --tpr_file 'threshold/RingID_old.pt'
#  python aa_eva_attack.py --distortion_list 'WAVEs_regen_use.json' --root '/work/maopy1998/RingID/' --attacked_path 'regen_positive_image/'  --watermark_model 'RingID' --filename 'RingID-adv.txt' --output_path 'last_output/' --tpr_file 'threshold/RingID_old.pt'
# python aa_eva_attack.py --distortion_list 'WAVEs_adv_use.json' --root '/work/maopy1998/RingID_16bit/' --attacked_path 'adv_positive_image/'  --watermark_model 'RingID' --filename 'RingID-high-adv.txt' --output_path 'last_output/' --tpr_file 'threshold/RingID.pt'
#  python aa_eva_attack.py --distortion_list 'WAVEs_regen_use.json' --root '/work/maopy1998/RingID_16bit/' --attacked_path 'regen_positive_image/'  --watermark_model 'RingID' --filename 'RingID-high-adv.txt' --output_path 'last_output/' --tpr_file 'threshold/RingID.pt'

# python aa_eva_attack.py --distortion_list 'WAVEs_regen_use2.json' --root '/work/maopy1998/RingID_16bit/' --attacked_path 'regen_positive_image/'  --watermark_model 'RingID' --filename 'RingID-high-adv.txt' --output_path 'last_output/' --tpr_file 'threshold/RingID.pt'

 # python aa_eva_attack.py --distortion_list 'WAVEs_adv_use.json' --root '/work/maopy1998/Gaussian-shading-16bit/' --attacked_path 'adv_positive_image/'  --watermark_model 'Gaussian-shading-general' --filename 'Gaussain-high-adv.txt' --output_path 'last_output/'  --hw_copy 2
#  python aa_eva_attack.py --distortion_list 'WAVEs_regen_use.json' --root '/work/maopy1998/Gaussian-shading-16bit/' --attacked_path 'regen_positive_image/'  --watermark_model 'Gaussian-shading-general' --filename 'Gaussain-high-adv.txt' --output_path 'last_output/'  --hw_copy 2

# python aa_eva_attack.py --distortion_list 'WAVEs_identification1.json' --root '/work/maopy1998/SuperShad_mean_12_5/' --attacked_path 'WAVE_positive_image/'  --watermark_model 'Super-shading' --filename 'Super-shading-wave_test.txt' --output_path 'last_output/' --hw_copy 2 --tpr_file 'threshold/super-shad.pt'

# python aa_eva_attack.py --distortion_list 'stirmark_all.json' --root '/work/maopy1998/tree-ring-ring/' --attacked_path 'pos_stirmark_image/'  --watermark_model 'tree-ring' --filename 'tree-ring-ring.txt' --output_path 'last_output/' --tpr_file 'threshold/tree-ring-ring.pt'
