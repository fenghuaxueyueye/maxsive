from schema.watermark import watermark
# from findpeaks import findpeaks
# from sklearn.decomposition import PCA
import torchvision
import numpy as np
import torch
import math
from torchvision.transforms.functional import resize
import os
from tqdm import trange

class shuffler():
    def __init__(self , hc, hw ) :
        self.hw = hw
        self.hc = hc
            
    def shuffle_key_gen(self):
        shuffle_keys = []
        for i in range(self.hc *self.hw * self.hw ):
            shuffle_keys.append( torch.randperm( 1* int(64/self.hw)* int(64/self.hw)* int(4/self.hc) ) )
        return shuffle_keys    
    def shuffle(self , z , shuffle_keys):
        all_z = []
        z_shape = z.shape
        z = z.reshape( -1)

        for i in range(self.hc):
            c = []
            for j in range(self.hw):
                w = []
                for k in range(self.hw):
                    w.append( z.clone()[shuffle_keys[i*self.hw*self.hw + j*self.hw + k ]].reshape(z_shape) )
                c.append(torch.concat(w ,axis = 3))
            
            all_z.append(torch.concat(c ,axis = 2) )
         
        return torch.concat(all_z , axis = 1 )
    def unshuffle(self , shuffled_z , shuffle_keys):
        
        ch_stride = 4 // self.hc
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.hc
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(shuffled_z, tuple(hw_list), dim=3), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(ch_list), dim=1), dim=0)
        unshuffled_z = []

        for i in range(self.hc):
            for j in range(self.hw):
                for k in range(self.hw):
                    
                    unshuf_order = torch.zeros( 1* int(64/self.hw)* int(64/self.hw)* int(4/self.hc) ).long()#.int()
                    unshuf_order[shuffle_keys[i*self.hw*self.hw + j*self.hw + k ]] = torch.arange( 1* int(64/self.hw)* int(64/self.hw)* int(4/self.hc) )
                    temp_shuffled_z = split_dim3[i*self.hw*self.hw + j*self.hw + k ].clone().reshape(-1)
                    unshuffled_z.append(temp_shuffled_z[unshuf_order].reshape(1, int(4/self.hc) ,int(64/self.hw) ,int(64/self.hw) ) )

        return  unshuffled_z   


class hough_transform:
    def __init__(self , x , y , dim , interval):
        self.dim = dim
        self.x = x
        self.y = y
        self.interval = interval
        self.init_map(dim , interval)
    def init_map( self ,  dim , interval ):
        self.degree_list = [interval* x for x in range(int(180/interval))]
        self.mask = torch.zeros( len(self.degree_list) ,dim , dim)
        for i , degree in enumerate(self.degree_list):
            for r in range(int(dim/2*(2)**0.5)):
                x_r = int( r*math.cos( math.radians( degree) ) )
                y_r = int( r*math.sin(  math.radians(degree )) )
                if abs(x_r) > int(dim/2)-1 or abs(y_r) > int(dim/2)-1:
                    break

                self.mask[ i  , self.x + x_r ,  self.y+y_r] =1
                self.mask[ i  , self.x - x_r , self.y-y_r ] =1
        self.w = self.mask.sum(dim=(1,2))
    def detect(self , x , include_angle , pre_angle):
        score = (x[None,:]*self.mask).sum(dim =(1,2)) / self.w
        x_score = []
        x_degree = []
        gap = int( include_angle/ self.interval)
        for i in range(score.shape[0]):

            x_degree.append((self.degree_list[i] ,self.degree_list[(i+gap)% score.shape[0]] ))
            x_score.append(score[i] + score[ (i+gap)% score.shape[0] ] )
        rotated_degree = self.degree_list[torch.tensor(x_score).topk(1)[1]]-pre_angle
        if_scale = self.detect_scale(x , include_angle , rotated_degree  )

        detection_result = {
            'rotated_degree' : rotated_degree,
            'if_scale' : if_scale,
            'x_degree' : x_degree[torch.tensor(x_score).topk(1)[1]],
            'value' : torch.tensor(x_score).topk(1)[0].item()
        }
        
        return detection_result #rotated_degree, x_degree[torch.tensor(x_score).topk(1)[1]],if_scale
    def detect_scale(self , ifft_zt , include_angle, predict_angle ):
        all_hist = []
        for boundwide in [-2 , -1 ,0 ,1 ,2]:
            degree_list = [ predict_angle-include_angle+boundwide ,predict_angle+boundwide ]
            mask = torch.zeros( int(64/2*(2)**0.5) ,64 , 64)
            # degree_list = [5* x for x in range(int(180/5))]
            for r in range(int(64/2*(2)**0.5)):
                for i , degree in enumerate( degree_list):
                    x_r = int( r*math.cos( math.radians( degree) ) )
                    y_r = int( r*math.sin(  math.radians(degree )) )
                    if abs(x_r) > int(64/2)-1 or abs(y_r) > int(64/2)-1:
                        break
                
                    mask[ r  , 32 + x_r ,  32+y_r] =1
                    mask[ r  , 32 - x_r , 32-y_r ] =1
            test = abs(ifft_zt) *mask
            r_strenge = test.reshape(45,-1).sum(axis = 1) / mask.reshape(45,-1).sum(axis=1)
            if r_strenge[16] > 50 and r_strenge[16] >r_strenge[17]  and r_strenge[16] >r_strenge[18] :
                all_hist.append( False)
            all_hist.append(True)
        if False in all_hist:
            return False
        return True

        
    def detect_scale_old(self , ifft_zt , include_angle, predict_angle ):
        degree_list = [ predict_angle-include_angle ,predict_angle ]
        mask = torch.zeros( int(64/2*(2)**0.5) ,64 , 64)
        # degree_list = [5* x for x in range(int(180/5))]
        for r in range(int(64/2*(2)**0.5)):
            for i , degree in enumerate( degree_list):
                x_r = int( r*math.cos( math.radians( degree) ) )
                y_r = int( r*math.sin(  math.radians(degree )) )
                if abs(x_r) > int(64/2)-1 or abs(y_r) > int(64/2)-1:
                    break
            
                mask[ r  , 32 + x_r ,  32+y_r] =1
                mask[ r  , 32 - x_r , 32-y_r ] =1
        test = abs(ifft_zt) *mask
        r_strenge = test.reshape(45,-1).sum(axis = 1) / mask.reshape(45,-1).sum(axis=1)
        if r_strenge[16] > 50 and r_strenge[16] >r_strenge[17]  and r_strenge[16] >r_strenge[18] :
            return False
        return True





class MaXsive(watermark):
    def __init__(self,  args   ):
        self.ch = args.channel_copy
        self.hw = args.hw_copy
        self.template_c = args.template_c
        self.shuffler = shuffler(self.ch , self.hw)
        self.barlett_window2D = self.creat_2d_window(64)
        self.latentlength = 4 * 64 * 64
        self.line_detection = hough_transform(32,32,64 , 5)
        self.distant_func = args.distant_func
        self.diffusion_bit = args.diffusion_bit
        
        if args.tpr_file is not None:
            self.data = torch.load(args.tpr_file)
            self.data =self.data.sort().values
            self.threshold = self.data[ - int(len(self.data)*0.001) ].item()
            
    def creat_2d_window(self , dim):
        window1d = torch.bartlett_window(dim)
        window2d = torch.sqrt(torch.outer(window1d,window1d))
        return window2d


    def watermark_injection(self):
        w = torch.randn( 1 , 4 // self.ch , 64 // self.hw , 64 // self.hw ).half()
        w = w/w.std()-w.mean()
        ### generate key 2
        shuffle_key = self.shuffler.shuffle_key_gen()
        z = self.shuffler.shuffle(w , shuffle_key)

        return z , {'z' : z ,  'keys': [w, shuffle_key ] }

    def k2_decode(self , z , shuffle_key):
        
        return self.shuffler.unshuffle(z , shuffle_key)

    
    def voting(self , zs) -> 'w':

        vote_z = torch.mean(torch.concat(zs), dim=0).clone()
        
        return vote_z
    
    def template_restore(self , z ,include_angle = 45 , pre_angle =135 ):
        z_fft = torch.fft.fftshift(torch.fft.fft2(z.float() ), dim=(-1, -2))
        ifft_zt= z_fft[0 , self.template_c].cpu() * self.barlett_window2D
        detection_result = self.line_detection.detect(abs(ifft_zt) , include_angle , pre_angle)
        degree , _ ,if_scale = detection_result['rotated_degree'], detection_result['x_degree'], detection_result['if_scale']
        
        if degree< 0:
            degree += 180
        print(degree ,'--rotated')
        if if_scale and degree != 0:
            scale = (math.sin(abs(degree%90)/180*math.pi) + math.cos(abs(degree%90)/180*math.pi)) 
            print('scale : ' , scale)
            
            z = torchvision.transforms.functional.resize(z ,  int(64/scale)) 
            z = torchvision.transforms.functional.resize(torchvision.transforms.functional.pad(z ,  int(  (64-z.shape[2])/2 )) ,64)
        return torchvision.transforms.functional.rotate(z , -degree)#
    def load_watermark_info(self , data):
        return data['keys']
        # return data
    
    def recover_watermark(self, z, data, rotation_restore=True):
        keys = self.load_watermark_info(data)
        if rotation_restore:
            z = self.template_restore(z)

        rotate_zs = self.k2_decode(z, keys[1])
        vote_rotate_z = self.voting(rotate_zs)
        return vote_rotate_z.reshape(1, -1), keys[0].reshape(1, -1)

    def bit_accuracy(self, z, data, rotation_restore=True):
        recovered_w, target_w = self.recover_watermark(z, data, rotation_restore=rotation_restore)
        recovered_bits = (recovered_w > 0).to(torch.int)
        target_bits = (target_w > 0).to(torch.int)

        bit_acc = (recovered_bits == target_bits).float().mean().item()
        return {
            'bit_acc': bit_acc,
            'ber': 1.0 - bit_acc,
        }
    
    def detection(self ,  z , data ,rotation_restore = True ):
        vote_rotate_z, w = self.recover_watermark(z, data, rotation_restore=rotation_restore)

        w = w.cuda()
        vote_rotate_z = vote_rotate_z.cuda()
        if self.distant_func == 'corr':
            cor1 = torch.corrcoef( torch.concat( [ w , vote_rotate_z]) )
            distant = abs(cor1[0,1].item() )
        if self.distant_func == 'cos':
            distant = torch.nn.functional.cosine_similarity(vote_rotate_z,w).item()
        if self.distant_func == 'mse':
            distant =  -torch.nn.functional.mse_loss( vote_rotate_z,w).item()
        if self.distant_func == 'l1':
            distant = -abs(vote_rotate_z - w ).mean().item()
        # print( 'corr' , distant)
        # return distant
        if distant > self.threshold:
            return 1
        return 0



class identification_MaXsive(MaXsive):
    def __init__(self , args):
        self.user_pool_path = args.user_pool_path
        self.user_pool = os.listdir(args.user_pool_path)
        # print(self.user_pool , '-----------')
        super().__init__(args)
    def identification(self ,  z , true_label  , rotation_restore = True):
        
        user_list  = []
        user_posibility = []
        if rotation_restore:
            z = self.template_restore(z )
            
        for i in trange(len(self.user_pool)):
            data = torch.load(self.user_pool_path + self.user_pool[i])
            user_list.append(int(self.user_pool[i].split('.')[0]))

            keys = self.load_watermark_info(data)
            
    
            rotate_zs = self.k2_decode( z , keys[1]  )
            vote_rotate_z = self.voting(rotate_zs)
    #         print(vote_rotate_z[:3].shape)
    
            w = keys[0].reshape(1,-1).cuda()
            
            vote_rotate_z = vote_rotate_z.reshape(1,-1).cuda()
            if self.distant_func == 'corr':
                cor1 = torch.corrcoef( torch.concat( [ w , vote_rotate_z]) )
                distant = abs(cor1[0,1].item() )
            if self.distant_func == 'cos':
                distant =  torch.nn.functional.cosine_similarity(vote_rotate_z,w).item()
            if self.distant_func == 'mse':
                distant =  -torch.nn.functional.mse_loss( vote_rotate_z,w).item()
            if self.distant_func == 'l1':
                distant = -abs(vote_rotate_z - w ).mean().item()
            user_posibility.append(distant)
        user_posibility = torch.tensor(user_posibility)
        print('predict user : ' , user_list[torch.argmax(user_posibility)] , 'true user : ' ,true_label )
        if user_list[torch.argmax(user_posibility)]==true_label:
            return True
        
        return False
            
