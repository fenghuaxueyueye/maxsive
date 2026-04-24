import torch
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
from schema.watermark import watermark
from models.binary_converter import bin2dec , dec2bin
import os
from tqdm import trange


           
        

class Gaussian_Shading_chacha(watermark):
    def __init__(self, args):
        self.fpr = args.fpr
        self.ch = args.channel_copy
        self.hw = args.hw_copy
        self.nonce = None
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.user_number = args.user_number

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.denominator =  2
        self.ppf = [norm.ppf(j / self.denominator) for j in range(int(self.denominator) + 1)]
        self.tau_onebit = None
        self.tau_bits = None
        
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw) * args.l
        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * self.user_number
            if fpr_onebit <= self.fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= self.fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength
        print(self.tau_onebit)        
        print(self.tau_bits ,'identification')
        
    def stream_key_encrypt(self, sd):
        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        print(m_bit.shape)
        return m_bit

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.cuda()
    def watermark_injection(self):
        w  = self.create_watermark_and_return_w()
        data = {
            'z' : w,
            'key' : self.key,
            'nonce' : self.nonce, 
            'watermark' : self.watermark
        }
        return w, data 
    def create_watermark_and_return_w(self):
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda() ###---------
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
        w = self.truncSampling(m)
        return w

    def stream_key_decrypt(self, reversed_m , key , nonce):
        cipher = ChaCha20.new(key=key, nonce=nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64).to(torch.uint8)
        return sd_tensor.cuda()

    def voting(self,watermark_r):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_w  , key ,nonce , watermark):
        print('eva-cha')
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy() , key ,nonce)
        # print('reversed_sd---' , reversed_sd)
        reversed_watermark = self.voting(reversed_sd)
        correct = (reversed_watermark == watermark).float().mean().item()
        print('correct ' , correct , 'threshold :' ,self.tau_onebit )
        if correct > self.tau_onebit:
            return 1
        return 0
    
    def detection(self ,  z , data ):
        key , nonce , watermark = self.load_watermark_info(data)
        return self.eval_watermark(z , key , nonce , watermark)

    def load_watermark_info(self , data):
        return  data['key'] ,  data['nonce'] ,data['watermark']
    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count

class identification_Gaussian_Shading(Gaussian_Shading_chacha):
    def __init__(self , args  ):
        super().__init__(args)
        self.user_pool_path = args.user_pool_path
        self.user_pool = os.listdir(args.user_pool_path)
        # print(self.user_pool , '-----------')
            
    def binary(self, x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    
    def create_watermark_and_return_w(self , user_id):
        self.watermark = self.user_w[user_id]
        # print(self.watermark)
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
        print(m.shape , '------------------')
        w = self.truncSampling(m)
        data = {
            'z' : w,
            'key' : self.key,
            'nonce' : self.nonce, 
            'watermark' : self.watermark
        }
        return w , data
    def watermark_injection(self , user_id):
        w ,data = self.create_watermark_and_return_w(user_id)
        return w, data 
    def identification(self ,  reversed_w , true_label ):
        user_list  = []
        user_posibility = []
        
        reversed_m = (reversed_w > 0).int()
        
        for i in trange(len(self.user_pool)):
            data = torch.load(self.user_pool_path + self.user_pool[i])
            
            user_list.append(int(self.user_pool[i].split('.')[0]))
            key , nonce , watermark = self.load_watermark_info(data)
            
            # print('mmm123' , reversed_m.shape , reversed_m.dtype)
            # print(reversed_m)
            reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().int().numpy() , key ,nonce)
            reversed_watermark = self.voting(reversed_sd)
            # print(reversed_watermark.shape)
            correct = (reversed_watermark == watermark).float().mean().item()
            user_posibility.append(correct)
            # print('bit acc : ' ,correct )
        user_posibility = torch.tensor(user_posibility)
        print('predict user : ' , user_list[torch.argmax(user_posibility)] , 'true user : ' ,true_label )
        
        if user_list[torch.argmax(user_posibility)]==true_label:
            return True
        
        return False  

class Gaussian_Shading_general_no_cha(watermark):
    def __init__(self, args):
        self.fpr = args.fpr
        self.l = args.l
        self.ch = args.channel_copy
        self.hw = args.hw_copy
        self.nonce = None
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)
        self.user_number = args.user_number

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.denominator =  2 ** self.l 
        self.ppf = [norm.ppf(j / self.denominator) for j in range(int(self.denominator) + 1)]
        self.tau_onebit = None
        self.tau_bits = None
        
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw) * args.l
        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * self.user_number
            if fpr_onebit <= self.fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= self.fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength
        print(self.tau_onebit)        
        print(self.tau_bits ,'identification')
    def stream_key_encrypt(self, sd):
        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def truncSampling(self, message):
        z = np.zeros(self.latentlength ,dtype=np.float16)
        
        for i in range(self.latentlength):
            dec_mes = message[i]
            dec_mes = bin2dec( torch.tensor(dec_mes ,dtype = torch.int64) , self.l)
            dec_mes = int(dec_mes)
            
            z[i] = truncnorm.rvs(self.ppf[dec_mes], self.ppf[dec_mes + 1])
            # print(dec_mes ,self.ppf[dec_mes], self.ppf[dec_mes + 1])
            # print('z i ' , z[i] , z.dtype)
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.cuda()
    def reverse_truncSampling(self , z):
        m = torch.zeros(z.shape)
        for i in range(len(self.ppf)  ):
            m[z > self.ppf[i] ] = i
        m = dec2bin(m.int() , self.l )
        # todo
        return m
    def watermark_injection(self):
        w = self.create_watermark_and_return_w()
        data = {
            'z' : w,
            'key' : self.key,
            'nonce' : self.nonce, 
            'watermark' : self.watermark
        }
        return w, data
    def create_watermark_and_return_w(self):
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw  , self.l ]).cuda()
        self.key = torch.randint(0, 2, [1, 4, 64 , 64  , self.l ]).cuda()
        
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw,1)
        m = ((sd + self.key) % 2).flatten().cpu().numpy()
        w = self.truncSampling(m.reshape( -1 , self.l))
        return w

    def stream_key_decrypt(self, reversed_m , key , nonce):
        cipher = ChaCha20.new(key=key, nonce=nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64 ,self.l ).to(torch.uint8)
        return sd_tensor.cuda()

    def voting(self,watermark_r):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        # print(watermark_r.shape)
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        # print(split_dim1.shape)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_w  , key ,nonce , watermark):
        print('eva-cha')
        reversed_m = self.reverse_truncSampling(reversed_w )
        # print('mmm123' , reversed_m.shape , reversed_m.dtype)
        # print(reversed_m)
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().int().numpy() , key ,nonce)
        reversed_watermark = self.voting(reversed_sd)
        # print(reversed_watermark.shape)
        correct = (reversed_watermark == watermark).float().mean().item()
        print('bit acc : ' ,correct )
        if correct > self.tau_onebit:
            return 1
        return 0
         
    
    def detection(self ,  z , data ):
        key , nonce , watermark = self.load_watermark_info(data)
        return self.eval_watermark(z , key , nonce , watermark)

    def load_watermark_info(self , data):
        return  data['key'] ,  data['nonce'] ,data['watermark']
    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count
        
class identification_Gaussian_Shading_no_cha(Gaussian_Shading_general_no_cha):
    def __init__(self , args  ):
        super().__init__(args)
        self.user_pool_path = args.user_pool_path
        self.user_pool = os.listdir(args.user_pool_path)
    def identification(self ,  reversed_w , true_label ):
        user_list  = []
        user_posibility = []
        reversed_m = self.reverse_truncSampling(reversed_w )
        
         
        for i in trange(len(self.user_pool)):
            data = torch.load(self.user_pool_path + self.user_pool[i])
            
            user_list.append(int(self.user_pool[i].split('.')[0]))
            key , nonce , watermark = self.load_watermark_info(data)
            # print(key)
            # print('mmm123' , reversed_m.shape , reversed_m.dtype)
            # print(reversed_m)
            reversed_sd = (reversed_m.cuda() + key) % 2
            reversed_watermark = self.voting(reversed_sd)
            # print(reversed_watermark.shape)
            correct = (reversed_watermark == watermark).float().mean().item()
            user_posibility.append(correct)
            # print('bit acc : ' ,correct )
        user_posibility = torch.tensor(user_posibility)
        print('predict user : ' , user_list[torch.argmax(user_posibility)] , 'true user : ' ,true_label )
        
        if user_list[torch.argmax(user_posibility)]==true_label:
            return True
        
        return False  