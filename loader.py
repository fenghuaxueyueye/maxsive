import torch
from dataset.datasets import ImageNet1000_prompt
from torch.utils.data import DataLoader
from models import TreeRing,Gaussian_Shading_chacha,MaXsive,RingID,identification_TreeRing,identification_Gaussian_Shading,identification_MaXsive,identification_Gaussian_Shading,identification_RingID
from attacks.regen import VAEWMAttacker, DiffWMAttacker
from attacks.regen_pipe import ReSDPipeline

def load_gen_model(name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'SD21':
        from inverse_stable_diffusion import InversableStableDiffusionPipeline
        from diffusers import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
        pipe = InversableStableDiffusionPipeline.from_pretrained(
                'stabilityai/stable-diffusion-2-1-base',
                scheduler=scheduler,
                torch_dtype=torch.float16,
                revision='fp16',
        )
        pipe.safety_checker = None
        pipe = pipe.to(device)

    if name == 'watermarkSD21':
        ## inprocess template
        from inverse_watermark_stable_diffusion import InversableStableDiffusionPipeline
        from modified_DPMSolver import Modified_DPMSolverMultistepScheduler
        scheduler = Modified_DPMSolverMultistepScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
        pipe = InversableStableDiffusionPipeline.from_pretrained(
                'stabilityai/stable-diffusion-2-1-base',
                scheduler=scheduler,
                torch_dtype=torch.float16,
                revision='fp16',
        )
        pipe.safety_checker = None
        pipe = pipe.to(device)
    return pipe

def load_dataset( args):
    start = args.start_num
    end = args.end_num

    #### load dataset
    if args.dataset == 'ImageNet':
        dataset = ImageNet1000_prompt( 'ImageNet_label.json' , args.num_of_imgs , start_num = start , end_num = end  )
        dataloader = DataLoader( dataset , batch_size = args.batch_size, shuffle=args.shuffle )

    return dataloader

def load_identification_watermark(args):

    if args.watermark_model == 'tree-ring':
        return identification_TreeRing(args)
    if args.watermark_model == 'Gaussian-shading':
        return identification_Gaussian_Shading(args)
    if args.watermark_model == 'Super-shading':
        return identification_MaXsive(args)
    if args.watermark_model =='RingID':
        return identification_RingID(args)



def load_watermark(args):

    if args.watermark_model == 'tree-ring':
        return TreeRing(args)
    if args.watermark_model == 'Gaussian-shading':
        return Gaussian_Shading_chacha(args)
    if args.watermark_model == 'MaXsive':
        return MaXsive(args)
    if args.watermark_model =='RingID':
        return RingID(args)

def load_attacker(args):
    if args.attack == 'regen_VAE':
        return VAEWMAttacker( args.vae_model , strength = args.strength )
    
    if args.attack == 'regen_diff':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipe = ReSDPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16"
        )
        pipe.to(device)
        return DiffWMAttacker(pipe , noise_step=args.strength  , is_P = False)
    
    if args.attack == 'regen_diffp':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipe = ReSDPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16"
        )
        pipe.to(device)

        return DiffWMAttacker(pipe ,  noise_step=args.strength  , is_P = True)
        


