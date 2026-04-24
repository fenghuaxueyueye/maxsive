from schema.watermark import watermark
from torchvision import transforms
import numpy as np
import torch
import itertools
import os
from tqdm import trange

device = 'cuda' if torch.cuda.is_available() else 'cpu'
RADIUS = 14
RADIUS_CUTOFF = 3
ANCHOR_X_OFFSET = 0     
ANCHOR_Y_OFFSET = 0     # 1 = not correct, 0 = correct
USE_ROUNDER_RING = True

HETER_WATERMARK_CHANNEL = [0]
RING_WATERMARK_CHANNEL = [3]
WATERMARK_CHANNEL = sorted(HETER_WATERMARK_CHANNEL + RING_WATERMARK_CHANNEL)

def fft(input_tensor):
    assert len(input_tensor.shape) == 4
    return torch.fft.fftshift(torch.fft.fft2(input_tensor), dim = (-1, -2))

def ifft(input_tensor):
    assert len(input_tensor.shape) == 4
    return torch.fft.ifft2(torch.fft.ifftshift(input_tensor, dim = (-1, -2)))

def get_distance(tensor1, tensor2, mask, p =1 , mode = 'complex', channel_min=1, channel = WATERMARK_CHANNEL):
    if tensor1.shape != tensor2.shape:
        raise ValueError(f'Shape mismatch during eval: {tensor1.shape} vs {tensor2.shape}')
    if mode not in ['complex', 'real', 'imag']:
        raise NotImplemented(f'Eval mode not implemented: {mode}')
    
    if not channel_min:
        if p == 1:
            # a faster implementation for l1 distance
            if mode == 'complex':
                return torch.mean(torch.abs(tensor1[0][channel] - tensor2[0][channel])[mask]).item()
            if mode == 'real':
                return torch.mean(torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)[mask]).item()
            if mode == 'imag':
                return torch.mean(torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)[mask]).item()
        else:
            if mode == 'complex':
                return torch.norm(torch.abs(tensor1[0][channel][mask] - tensor2[0][channel][mask]), p = p).item() / torch.sum(mask)
            if mode == 'real':
                return torch.norm(torch.abs(tensor1[0][channel][mask].real - tensor2[0][channel][mask].real), p = p).item() / torch.sum(mask)
            if mode == 'imag':
                return torch.norm(torch.abs(tensor1[0][channel][mask].imag - tensor2[0][channel][mask].imag), p = p).item() / torch.sum(mask)
    else:
        # argmin TODO: normalize
        if  len(RING_WATERMARK_CHANNEL) > 1 and len(HETER_WATERMARK_CHANNEL) > 0:
            ring_channel_idx_list = [idx for idx, c_id in enumerate(WATERMARK_CHANNEL) if c_id in RING_WATERMARK_CHANNEL]
            heter_channel_idx_list = [idx for idx, c_id in enumerate(WATERMARK_CHANNEL) if c_id in HETER_WATERMARK_CHANNEL]
            if mode == 'complex':
                diff = torch.abs(tensor1[0][channel] - tensor2[0][channel])  # [c, h, w]
            elif mode == 'real':
                diff = torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)  # [c, h, w]
            elif mode == 'imag':
                diff = torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)  # [c, h, w]
            l1_list = []
            num_items = []
            for c_idx in range(len(mask)):
                mask_c = torch.zeros_like(mask)
                mask_c[c_idx] = mask[c_idx]
                l1_list.append(torch.mean(diff[mask_c]).item())
                num_items.append(torch.sum(mask_c).item())
            total = 0
            num = 0
            for ring_channel_idx in ring_channel_idx_list:
                total += l1_list[ring_channel_idx] * num_items[ring_channel_idx]
                num += num_items[ring_channel_idx]
            ring_channels_mean = total / num
            return min(ring_channels_mean, min([l1_list[idx] for idx in heter_channel_idx_list]))
        elif len(RING_WATERMARK_CHANNEL) == 1 and len(HETER_WATERMARK_CHANNEL) > 0:
            if mode == 'complex':
                diff = torch.abs(tensor1[0][channel] - tensor2[0][channel])  # [c, h, w]
            elif mode == 'real':
                diff = torch.abs(tensor1[0][channel].real - tensor2[0][channel].real)  # [c, h, w]
            elif mode == 'imag':
                diff = torch.abs(tensor1[0][channel].imag - tensor2[0][channel].imag)  # [c, h, w]
            l1_list = []
            for c_idx in range(len(mask)):
                mask_c = torch.zeros_like(mask)
                mask_c[c_idx] = mask[c_idx]
                l1_list.append(torch.mean(diff[mask_c]).item())
            return min(l1_list)
        else:
            raise NotImplementedError


def make_Fourier_ringid_pattern(
        device,
        key_value_combination, 
        no_watermark_latents, 
        radius,
        radius_cutoff, 
        ring_watermark_channel, 
        heter_watermark_channel, 
        heter_watermark_region_mask=None,
        ring_width = 1, 
        ):
    if ring_width != 1:
        raise NotImplementedError(f'Proposed watermark generation only implemented for ring width = 1.')

    if len(key_value_combination) != (RADIUS - RADIUS_CUTOFF):
        raise ValueError('Mismatch between #key values and #slots')
    
    shape = no_watermark_latents.shape
    if len(shape) != 4:
        raise ValueError(f'Invalid shape for initial latent: {shape}')
    
    latents_fft = fft(no_watermark_latents)
    # watermarked_latents_fft = copy.deepcopy(latents_fft)
    watermarked_latents_fft = torch.zeros_like(latents_fft)

    radius_list = [this_radius for this_radius in range(radius, radius_cutoff, -1)]

    # put ring
    for radius_index in range(len(radius_list)):
        this_r_out = radius_list[radius_index]
        this_r_in = this_r_out - ring_width
        mask = torch.tensor(ring_mask(size = shape[-1], r_out = this_r_out, r_in = this_r_in)).to(device).to(torch.float64)  # sector_idx default to -1
        for batch_index in range(shape[0]):
            for channel_index in range(len(ring_watermark_channel)):
                watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].real = (1 - mask) * watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].real + mask * key_value_combination[radius_index][channel_index]
                watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].imag = (1 - mask) * watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].imag + mask * key_value_combination[radius_index][channel_index]

    # put noise or zeros
    if len(heter_watermark_channel) > 0:
        assert len(heter_watermark_channel) == len(heter_watermark_region_mask)
        heter_watermark_region_mask = heter_watermark_region_mask.to(torch.float64)
        w_type = 'noise'
        
        if w_type == 'noise':
            w_content = fft(torch.randn(*shape, device = device))  # [N, c, h, w]
        elif w_type == 'zeros':
            w_content = fft(torch.zeros(*shape, device = device))  # [N, c, h, w]
        else:
            raise NotImplementedError
        
        for batch_index in range(shape[0]):
            for channel_id, channel_mask in zip(heter_watermark_channel, heter_watermark_region_mask):
                watermarked_latents_fft[batch_index, channel_id].real = \
                    (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].real + channel_mask * w_content[batch_index][channel_id].real
                watermarked_latents_fft[batch_index, channel_id].imag = \
                    (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].imag + channel_mask * w_content[batch_index][channel_id].imag

    return watermarked_latents_fft

class RounderRingMask:
    def __init__(self, size = 64, r_out = RADIUS, x_offset = ANCHOR_X_OFFSET, y_offset = ANCHOR_Y_OFFSET, mode = 'full'):
        assert size >= 3
        self.size = size
        self.r_out = r_out

        num_rings = r_out
        zero_bg_freq = torch.zeros(size, size)
        center = size // 2
        center_x, center_y = center + x_offset, center - y_offset

        ring_vector = torch.tensor([(200 - i*4) * (-1)**i for i in range(num_rings)])
        zero_bg_freq[center_x, center_y:center_y+num_rings] = ring_vector
        zero_bg_freq = zero_bg_freq[None, None, ...]
        self.ring_vector_np = ring_vector.numpy()

        res = torch.zeros(360, size, size)
        res[0] = zero_bg_freq
        for angle in range(1, 360):
            zero_bg_freq_rot = transforms.functional.rotate(zero_bg_freq, angle=angle)
            res[angle] = zero_bg_freq_rot

        res = res.numpy()
        self.pure_bg = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                values, count = np.unique(res[:, x, y],  return_counts=True)
                if len(count) > 2:
                    self.pure_bg[x, y] = values[count == max(count[values!=0])][0]
                elif len(count) == 2:
                    self.pure_bg[x, y] = values[values!=0][0]
        
    def get_ring_mask(self, r_out, r_in):
        """
            get mask from pure_bg
            sector_idx == -1, no sectors, get the whole ring
            sector_idx in [0, 1] for 2 effective sectors
        """
        assert r_out <= self.r_out
        if r_in - 1 < 0:
            right_end = 0  # None, to take the center
        else:
            right_end = r_in - 1
        cand_list = self.ring_vector_np[r_out-1:right_end:-1]
        mask = np.isin(self.pure_bg, cand_list)
        
        if self.size % 2:
            mask = mask[:self.size-1, :self.size-1]  # [64, 64]

        return mask


mask_obj = RounderRingMask(size=65, r_out=RADIUS, x_offset = ANCHOR_X_OFFSET, y_offset = ANCHOR_Y_OFFSET)
def ring_mask(size = 64, r_out = RADIUS, r_in = RADIUS_CUTOFF, x_offset = ANCHOR_X_OFFSET, y_offset = ANCHOR_Y_OFFSET, mode = 'full'):
    assert size == 64
    assert mode == 'full', f'not implemented mode {mode}'
    return mask_obj.get_ring_mask(r_out=r_out, r_in=r_in)

def generate_Fourier_watermark_latents(watermark_region_mask, watermark_channel, original_latents = None, watermark_pattern = None):
    
    #set_random_seed(seed)

    if original_latents is None:
        #original_latents = torch.randn(*shape, device = device)
        raise NotImplementedError('Original latents should be provided.')
    
    if watermark_pattern is None:
        raise NotImplementedError('Fourier watermark pattern should be provided.')

    # circular_mask = torch.tensor(ring_mask(size = original_latents.shape[-1], r_out = radius, r_in = radius_cutoff)).to(device)
    watermarked_latents_fft = torch.fft.fftshift(torch.fft.fft2(original_latents), dim = (-1, -2))

    # for channel in watermark_channel:
    #     watermarked_latents_fft[:, channel] = watermarked_latents_fft[:, channel] * ~circular_mask + watermark_pattern[:, channel] * circular_mask
    
    assert len(watermark_channel) == len(watermark_region_mask)
    for channel, channel_mask in zip(watermark_channel, watermark_region_mask):
        watermarked_latents_fft[:, channel] = watermarked_latents_fft[:, channel] * ~channel_mask + watermark_pattern[:, channel] * channel_mask
    
    return torch.fft.ifft2(torch.fft.ifftshift(watermarked_latents_fft, dim = (-1, -2))).real

class RingID(watermark):
    def __init__(self , args):
        self.ring_value_range = args.ring_value_range
        self.quantization_levels = args.quantization_levels
        self.Fourier_watermark_pattern_list , self.watermark_region_mask = self.make_all_mask(torch.randn( 1 , 4  , 64  , 64 ).cuda())
        # print(len(self.Fourier_watermark_pattern_list) , len(self.watermark_region_mask))
        self.ring_capacity = len(self.Fourier_watermark_pattern_list)
        # get heterogeneous watermark mask
        if len(HETER_WATERMARK_CHANNEL) > 0:
            single_channel_heter_watermark_mask = torch.tensor(
                    ring_mask(
                        size = 64, 
                        r_out = RADIUS, 
                        r_in = RADIUS_CUTOFF)  # TODO: change to whole mask
                    )
            self.heter_watermark_region_mask = single_channel_heter_watermark_mask.unsqueeze(0).repeat(len(HETER_WATERMARK_CHANNEL), 1, 1).to(device)
        if args.tpr_file is not None:
            self.data = torch.load(args.tpr_file)
            self.data =self.data.sort().values
            self.threshold = self.data[ - int(len(self.data)*0.001) ].item()
    def make_all_mask(self,z):

        sing_channel_ring_watermark_mask = torch.tensor(
                    ring_mask(
                        size = z.shape[-1], 
                        r_out = RADIUS, 
                        r_in = RADIUS_CUTOFF)
                    )
        
        # get heterogeneous watermark mask
        if len(HETER_WATERMARK_CHANNEL) > 0:
            single_channel_heter_watermark_mask = torch.tensor(
                    ring_mask(
                        size = z.shape[-1], 
                        r_out = RADIUS, 
                        r_in = RADIUS_CUTOFF)  # TODO: change to whole mask
                    )
            heter_watermark_region_mask = single_channel_heter_watermark_mask.unsqueeze(0).repeat(len(HETER_WATERMARK_CHANNEL), 1, 1).to(device)

        
        watermark_region_mask = []
        for channel_idx in WATERMARK_CHANNEL:
            if channel_idx in RING_WATERMARK_CHANNEL:
                watermark_region_mask.append(sing_channel_ring_watermark_mask)
            else:
                watermark_region_mask.append(single_channel_heter_watermark_mask)
        watermark_region_mask = torch.stack(watermark_region_mask).to(device)  # [C, 64, 64]
        
        print(watermark_region_mask.shape )
        
        single_channel_num_slots = RADIUS - RADIUS_CUTOFF
        key_value_list = [[list(combo) for combo in itertools.product(np.linspace(-self.ring_value_range, self.ring_value_range, self.quantization_levels).tolist(), repeat = len(RING_WATERMARK_CHANNEL))] for _ in range(single_channel_num_slots)]
        # print(key_value_list)
        key_value_combinations = list(itertools.product(*key_value_list))
        
        Fourier_watermark_pattern_list = [make_Fourier_ringid_pattern(device, list(combo), z, 
                                                                                        radius=RADIUS, radius_cutoff=RADIUS_CUTOFF,
                                                                                        ring_watermark_channel=RING_WATERMARK_CHANNEL, 
                                                                                        heter_watermark_channel=HETER_WATERMARK_CHANNEL,
                                                                                        heter_watermark_region_mask=heter_watermark_region_mask if len(HETER_WATERMARK_CHANNEL)>0 else None)
                                                                                        for _, combo in enumerate(key_value_combinations)]    
        ring_capacity = len(Fourier_watermark_pattern_list)
        print('ring_capacity' , ring_capacity)
        # print(Fourier_watermark_pattern_list)
        Fourier_watermark_pattern_list = [fft(ifft(Fourier_watermark_pattern).real) for Fourier_watermark_pattern in Fourier_watermark_pattern_list]
        
        for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
            # Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)) * args.time_shift_factor)
            Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)))

        
        return Fourier_watermark_pattern_list , watermark_region_mask

    
    def watermark_injection(self ):
        z = torch.randn( 1 , 4  , 64  , 64 ).cuda()
        key_index = np.random.choice( self.ring_capacity, size = 1, replace = True).tolist()[0]
        
        Fourier_watermark_latents = generate_Fourier_watermark_latents(
            original_latents = z, 
            watermark_pattern = self.Fourier_watermark_pattern_list[key_index],
            watermark_channel = WATERMARK_CHANNEL,
            watermark_region_mask = self.watermark_region_mask,
        )
        print(self.watermark_region_mask.shape)
        data = {
            'z'                     : z,
            'watermark_pattern'     : self.Fourier_watermark_pattern_list[key_index],
            'watermark_region_mask' : self.watermark_region_mask
        }
        return Fourier_watermark_latents.half(), data

        
    def detection(self ,  z , data):
        distance_per_gt = get_distance(data['watermark_pattern'] , fft(z.float()), 
                                                          data['watermark_region_mask'])
        # print('distant : ' ,-distance_per_gt)
        # return -distance_per_gt
        print(self.threshold ,-distance_per_gt  )
        if (-distance_per_gt) > self.threshold:
            return 1
        return 0
class identification_RingID(RingID):
    def __init__(self , args  ):
        super().__init__(args)
        self.user_pool_path = args.user_pool_path
        self.user_pool = os.listdir(args.user_pool_path)
        # print(self.user_pool , '-----------')
    def identification(self ,  reversed_w , true_label ):
        user_list  = []
        user_posibility = []
        for i in trange(len(self.user_pool)):
            data = torch.load(self.user_pool_path + self.user_pool[i])
            distance_per_gt = get_distance(data['watermark_pattern'] , fft(reversed_w.float()), 
                                                              data['watermark_region_mask'])
            user_list.append(int(self.user_pool[i].split('.')[0]))
            user_posibility.append(distance_per_gt)
            # print(distance_per_gt)
        user_posibility = torch.tensor(user_posibility)
        print('predict user : ' , user_list[torch.argmin(user_posibility)] , 'true user : ' ,true_label )
        
        if user_list[torch.argmin(user_posibility)]==true_label:
            return True
        
        return False         