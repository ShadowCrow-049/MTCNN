import torch
import numpy as np
from tools import nms
import Nets
from torchvision import transforms


class P_net_detector:
    def __init__(self, P_net_param="网络参数\P_net.pt", isCuda=True):
        self.isCuda = isCuda
        self.P_net = Nets.P_net()

        if self.isCuda:
            self.P_net.cuda()

        self.P_net.load_state_dict(torch.load(P_net_param))

        self.P_net.eval()

        self.m = transforms.ToTensor()

    def detect(self, image):

        w, h = image.size
        min_side = min(w, h)  # 以便做图像金字塔
        max_side=max(w,h)
        scale = 1  # 设置初始的缩放比例
        # alpha=(0.7 if(max_side)<1000 else 0.4)
        alpha=0.5
        boxes_ = []
        while min_side >= 12:
            img_data = self.m(image)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)  # 老版本的pytorch需要在第一个轴升一个维度作为批次
            cond_, position_offset_ ,landmark_offset_= self.P_net(img_data)
            _cond = cond_.cpu().detach()

            _position_offset = position_offset_.cpu().detach()
            boxes = nms(self.return_box(_cond, _position_offset, 0.6, scale), 0.5)  # (-1,5)

            boxes_.extend(boxes)

            # boxes_ = torch.cat((boxes_, boxes), 0)

            scale *= alpha
            _w = int(w * scale)
            _h = int(h * scale)
            image = image.resize((_w, _h))
            min_side = min(_w, _h)

        return np.array(boxes_)  # 这里返回的是做完金字塔后对所有框作NMS后的框，因为计算时间要少一些

    def return_box(self, cond, offset, c, scale):
        _cond = cond[0][0]
        cond_ = torch.nonzero(torch.gt(_cond, c)).float()

        real_box = torch.tensor([])

        x1 = offset[:, 0][torch.gt(cond[:, 0], c)].view(-1, 1).float()
        y1 = offset[:, 1][torch.gt(cond[:, 0], c)].view(-1, 1).float()
        x2 = offset[:, 2][torch.gt(cond[:, 0], c)].view(-1, 1).float()
        y2 = offset[:, 3][torch.gt(cond[:, 0], c)].view(-1, 1).float()

        real_box = torch.cat((real_box, (cond_[:, 1:] * 2 + x1 * 12) / scale), 1)
        real_box = torch.cat((real_box, (cond_[:, :1] * 2 + y1 * 12) / scale), 1)
        real_box = torch.cat((real_box, (cond_[:, 1:] * 2 + x2 * 12 + 12) / scale), 1)
        real_box = torch.cat((real_box, (cond_[:, :1] * 2 + y2 * 12 + 12) / scale), 1)
        real_box = torch.cat((real_box, cond[:, 0][torch.gt(cond[:, 0], c)].view(-1, 1)), 1)

        return real_box