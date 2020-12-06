import torch
import numpy as np
from tools import nms,convert_to_square
import Nets
from torchvision import transforms

class R_net_detector:
    def __init__(self,R_net_param="网络参数\R_net.pt",isCuda=True):
        self.isCuda=isCuda
        self.R_net = Nets.R_net()

        if self.isCuda:
            self.R_net.cuda()

        self.R_net.load_state_dict(torch.load(R_net_param))

        self.R_net.eval()

        self.m = transforms.ToTensor()

    def detect(self, image, P_net_boxes):
        if len(P_net_boxes)==0:
            return np.array([])
        _img_dataset = []  # 用于存放P网络输出的框对应在原图上的框
        _P_net_boxes = convert_to_square(P_net_boxes)  # 将P网络输出的框转化为正方形

        # 按照新的正方形在原图上抠图
        for _box in _P_net_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            # 将扣的新图传入到R网络中
            img_data = self.m(img)
            _img_dataset.append(img_data)
        img_dataset = torch.stack(_img_dataset)


        if self.isCuda:
            img_dataset = img_dataset.cuda()
        cond_, position_offset_ ,landmark_offset_= self.R_net(img_dataset)

        cond = cond_.cpu().detach()
        position_offset = position_offset_.cpu().detach()

        boxes_ = []
        real_box = torch.tensor([])
        _P_net_boxes=torch.tensor(_P_net_boxes)
        _x1 = _P_net_boxes[:, 0][torch.gt(cond[:, 0], 0.6)].view(-1, 1).float()

        _y1 = _P_net_boxes[:, 1][torch.gt(cond[:, 0], 0.6)].view(-1, 1).float()
        _x2 = _P_net_boxes[:, 2][torch.gt(cond[:, 0], 0.6)].view(-1, 1).float()
        _y2 = _P_net_boxes[:, 3][torch.gt(cond[:, 0], 0.6)].view(-1, 1).float()

        x1=position_offset[:, 0][torch.gt(cond[:, 0], 0.6)].view(-1, 1).float()
        y1=position_offset[:, 1][torch.gt(cond[:, 0], 0.6)].view(-1, 1).float()
        x2=position_offset[:, 2][torch.gt(cond[:, 0], 0.6)].view(-1, 1).float()
        y2=position_offset[:, 3][torch.gt(cond[:, 0], 0.6)].view(-1, 1).float()

        real_box = torch.cat((real_box, _x1+(_x2-_x1)*x1), 1)
        real_box = torch.cat((real_box, _y1+(_y2-_y1)*y1), 1)
        real_box = torch.cat((real_box, _x2+(_x2-_x1)*x2), 1)
        real_box = torch.cat((real_box, _y2+(_y2-_y1)*y2), 1)
        real_box = torch.cat((real_box, cond[:, 0][torch.gt(cond[:, 0], 0.6)].view(-1, 1)), 1)
        boxes = nms(real_box, 0.5)  # (-1,5)
        boxes_.extend(boxes)
        return np.array(boxes_)


