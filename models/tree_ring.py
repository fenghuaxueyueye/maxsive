from schema.watermark import watermark
from tqdm import trange
import numpy as np
import torch
import copy
import os


        


class TreeRing(watermark):
    def __init__(self , args ):
        self.w_mask_shape   = args.w_mask_shape
        self.w_channel      = args.w_channel
        self.w_pattern = args.w_pattern
        self.w_radius = args.w_radius
        self.w_pattern_const = args.w_pattern_const
        self.w_injection  = args.w_injection 
        if args.tpr_file is not None:
            self.data = torch.load(args.tpr_file)
            self.data =self.data.sort().values
            self.threshold = self.data[ - int(len(self.data)*0.001) ].item()
    def watermark_injection(self):
        z = torch.randn( 1 , 4  , 64  , 64 ).cuda()
        gt_patch = self.get_watermarking_pattern(z)
        watermarking_mask = self.get_watermarking_mask(z)
        init_latent_w = self.inject_watermark( z.clone(), watermarking_mask, gt_patch)
        data = {
            'z' : z,
            'gt_patch' : gt_patch,
            'watermarking_mask': watermarking_mask
        }
        return init_latent_w.half() , data
    def detection(self , z , data , w_measurement = 'l1_complex'):
        distant = self.eval_watermark(z , data ,w_measurement)
        print('distant : ' ,distant)
        # return distant
        # print('distant : ' ,distant)
        if (distant) > self.threshold:
            return 1
        return 0
    
    def load_watermark_info(self , data):
        return  data['gt_patch'] ,data['watermarking_mask'] 

    def eval_watermark(self , reversed_latents_no_w, data , w_measurement = 'l1_complex'):
        gt_patch , watermarking_mask = self.load_watermark_info(data)
        if 'complex' in w_measurement:
            print('to complex')
            reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
            target_patch = gt_patch
        elif 'seed' in w_measurement:
            reversed_latents_no_w_fft = reversed_latents_no_w
            target_patch = gt_patch
        else:
            NotImplementedError(f'w_measurement: {w_measurement}')

        if 'l1' in w_measurement:
            no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        else:
            NotImplementedError(f'w_measurement: {w_measurement}')

        return -no_w_metric


    ###------------------- not finish yet
    def circle_mask(self , size=64, r=10, x_offset=0, y_offset=0):
        # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
        x0 = y0 = size // 2
        x0 += x_offset
        y0 += y_offset
        y, x = np.ogrid[:size, :size]
        y = y[::-1]

        return ((x - x0)**2 + (y-y0)**2)<= r**2


    def get_watermarking_mask(self , init_latents_w):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

        if self.w_mask_shape == 'circle':
            np_mask = self.circle_mask(init_latents_w.shape[-1], r=self.w_radius)
            torch_mask = torch.tensor(np_mask).to(device)

            if self.w_channel == -1:
                # all channels
                watermarking_mask[:, :] = torch_mask
            else:
                watermarking_mask[:, self.w_channel] = torch_mask
        elif self.w_mask_shape == 'square':
            anchor_p = init_latents_w.shape[-1] // 2
            if self.w_channel == -1:
                # all channels
                watermarking_mask[:, :, anchor_p-self.w_radius:anchor_p+self.w_radius, anchor_p-self.w_radius:anchor_p+self.w_radius] = True
            else:
                watermarking_mask[:, self.w_channel, anchor_p-self.w_radius:anchor_p+self.w_radius, anchor_p-self.w_radius:anchor_p+self.w_radius] = True
        elif self.w_mask_shape == 'no':
            pass
        elif self.w_mask_shape == 'all':
            watermarking_mask = torch.ones(init_latents_w.shape, dtype=torch.bool).to(device)
        else:
            raise NotImplementedError(f'w_mask_shape: {self.w_mask_shape}')

        return watermarking_mask


    def get_watermarking_pattern(self ,gt_init ):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'seed_ring' in self.w_pattern:
            gt_patch = gt_init

            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.w_radius, 0, -1):
                tmp_mask = self.circle_mask(gt_init.shape[-1], r=i)
                tmp_mask = torch.tensor(tmp_mask).to(device)
                
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        elif 'seed_zeros' in self.w_pattern:
            gt_patch = gt_init * 0
        elif 'seed_rand' in self.w_pattern:
            gt_patch = gt_init
        elif 'rand' in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch[:] = gt_patch[0]
        elif 'zeros' in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        elif 'const' in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
            gt_patch += self.w_pattern_const
        elif 'ring' in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.w_radius, 0, -1):
                tmp_mask = self.circle_mask(gt_init.shape[-1], r=i)
                tmp_mask = torch.tensor(tmp_mask).to(device)
                
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

        return gt_patch


    def inject_watermark(self , init_latents_w, watermarking_mask, gt_patch):
        init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
        if self.w_injection == 'complex':
            init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
        elif self.w_injection == 'seed':
            init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
            return init_latents_w
        else:
            NotImplementedError(f'w_injection: {self.w_injection}')

        init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

        return init_latents_w


class identification_TreeRing(TreeRing):
    def __init__(self , args):
        self.user_pool_path = args.user_pool_path
        self.user_pool = os.listdir(args.user_pool_path)
        # print(self.user_pool , '-----------')
        super().__init__(args)
        
        
        
        
    def identification(self ,  reversed_latents_no_w , true_label , w_measurement = 'l1_complex'):
        user_list  = []
        user_posibility = []
        for i in trange(len(self.user_pool)):
            data = torch.load(self.user_pool_path + self.user_pool[i])
            user_list.append(int(self.user_pool[i].split('.')[0]))
            
            gt_patch , watermarking_mask = self.load_watermark_info( data)
            if 'complex' in w_measurement:
                reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
                target_patch = gt_patch
            elif 'seed' in w_measurement:
                reversed_latents_no_w_fft = reversed_latents_no_w
                target_patch = gt_patch
            else:
                NotImplementedError(f'w_measurement: {w_measurement}')
    
            if 'l1' in w_measurement:
                no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
            if 'l2' in w_measurement:
                no_w_metric = torch.abs( (reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask])*(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]) ).mean().item()
            else:
                NotImplementedError(f'w_measurement: {w_measurement}')
            user_posibility.append(no_w_metric)
            # print(user_posibility)
        user_posibility = torch.tensor(user_posibility)
        print('predict user : ' , user_list[torch.argmin(user_posibility)] , 'true user : ' ,true_label  , torch.argmin(user_posibility))
        # print(user_list)
        print( (-user_posibility).topk(10) )
        if user_list[torch.argmin(user_posibility)]==true_label:
            return True
        
        return False