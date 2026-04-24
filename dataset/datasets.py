import json
from torch.utils.data import Dataset



class ImageNet1000_prompt(Dataset):
    
    def __init__(self, csv_file , num_of_imgs, start_num = 0 , end_num = 999):
        
        self.prompts = []
        self.id = []
        # Open and read the JSON file
        with open(csv_file, 'r') as file:
            self.data = json.load(file)
        for i in range(start_num ,end_num+1 ):
            self.prompts += ['A photo of a ' + self.data[str(i)].split(',')[0]]*num_of_imgs
            self.id += [str(i)]*num_of_imgs
#             if i==1:
#                 print('check prompt : ',self.prompts)
#         self.prompts = self.prompts[:3]
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.id[idx], self.prompts[idx]