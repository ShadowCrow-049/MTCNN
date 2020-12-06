from torch.utils.data import Dataset,dataloader
import os
import numpy as np
import torch
import PIL.Image as pimg
from torchvision import  transforms


class getdatas(Dataset):
    def __init__(self,path):
        self.m =transforms.ToTensor()
        self.path=path
        self.dataset=[]
        self.dataset.extend(open(os.path.join(path,"positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())

    def __getitem__(self, index):
        strs=self.dataset[index].strip().split(" ")

        img_path = os.path.join(self.path, strs[0])
        cond=torch.Tensor([int(strs[1])])
        position_offset=torch.Tensor([float(strs[2]),float(strs[3]),float(strs[4]),float(strs[5])])
        landmark_offset=torch.Tensor([float(strs[6]),float(strs[7]),float(strs[8]),float(strs[9]),float(strs[10]),float(strs[11]),float(strs[12]),float(strs[13]),float(strs[14]),float(strs[15])])
        img_data=np.array(pimg.open(img_path))
        img_data=self.m(img_data)
        return img_data,cond,position_offset,landmark_offset
    def __len__(self):
        return len(self.dataset)

# if __name__=="__main__":
#     dataset=getdatas(r"D:\Pycharm\MTCNN数据集制作\训练数据样本\12")
#
#     data=DataLoader(dataset=dataset,batch_size=1,shuffle=True)
#     for i,(img_,cond_,offset_) in enumerate(data):
#         print(img_)
#         print(cond_)
#         print(offset_)
#         print("####")






