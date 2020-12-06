import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from Getdatas import getdatas
class Trainer:
    def __init__(self,net,save_para_path,dataset_path,iscuda=True):
        self.net=net
        self.save_para_path=save_para_path
        self.data_path=dataset_path
        self.iscuda=iscuda
        if self.iscuda:
            self.net.cuda()
        self.cond_lossfunc=nn.BCELoss()
        self.offset_lossfunc=nn.MSELoss()
        self.opt=optim.Adam(params=net.parameters(),lr=0.001)
        # 如果路径中有保存好的网络参数，则加载网络参数继续训练
        if os.path.exists(self.save_para_path):
            net.load_state_dict(torch.load(self.save_para_path))
    def train(self):
        facedataset=getdatas(self.data_path)
        # dataloader=DataLoader(facedataset,batch_size=512,shuffle=True,num_workers=4)
        dataloader=DataLoader(facedataset,batch_size=512,shuffle=True,num_workers=4)
        count = 0
        while True:
            for i,(img_data_,cond_,position_offset_,landmark_offset_) in enumerate(dataloader):
                if self.iscuda:
                    # img_data_=img_data_.float().cuda()
                    img_data_=img_data_.cuda()
                    cond_=cond_.cuda()
                    position_offset_=position_offset_.cuda()
                    landmark_offset_=landmark_offset_.cuda()

                    # 计算置信度的损失
                cond_output_,position_offset_output_,landmark_offset_output_=self.net(img_data_)

                cond_output=cond_output_.reshape(-1,1)

                # 部分样本不参与置信度损失的计算
                cond_mask=torch.lt(cond_,2) # 得到置信度小于2的掩码，若小于2掩码为1，大于等于2掩码为零
                cond=torch.masked_select(cond_,cond_mask)  # 根据cond_mask中将位置为1的对应于cond_中将置信度取出来
                cond_output=torch.masked_select(cond_output,cond_mask)

                cond_loss=self.cond_lossfunc(cond_output,cond)

                #计算建议框偏移量的损失
                # 负样本不参与偏移量损失的计算
                position_offset_mask=torch.gt(cond_,0)  # 得到置信度大于0的掩码，若小于等于掩码为0，大于0掩码为1
                position_offset=position_offset_[position_offset_mask[:,0]]
                position_offset_output=position_offset_output_[position_offset_mask[:,0]]
                position_offset_output=position_offset_output.reshape(-1,4)
                position_offset_loss=self.offset_lossfunc(position_offset_output,position_offset)


                # 计算五官偏移量的损失
                # 负样本不参与偏移量损失的计算
                landmark_offset_mask = torch.gt(cond_, 0)  # 得到置信度大于0的掩码，若小于等于掩码为0，大于0掩码为1
                landmark_offset = landmark_offset_[landmark_offset_mask[:, 0]]
                landmark_offset_output = landmark_offset_output_[landmark_offset_mask[:, 0]]
                landmark_offset_output = landmark_offset_output.reshape(-1, 10)
                landmark_offset_loss = self.offset_lossfunc(landmark_offset_output, landmark_offset)

                loss=cond_loss+position_offset_loss+landmark_offset_loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                print("loss:",loss.float().item(),"cond_loss:",cond_loss.float().item(),"position_offset_loss",position_offset_loss.float().item(),"landmark_offset_loss",landmark_offset_loss.float().item())
            count=count+1
            print("第{0}轮训练完毕".format(count))
            torch.save(self.net.state_dict(),self.save_para_path)
            print("保存成功")



















