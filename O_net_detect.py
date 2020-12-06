import torch
import numpy as np
from tools import nms,convert_to_square
import Nets
from torchvision import transforms

class O_net_detector:
    def __init__(self,O_net_param="网络参数\O_net.pt",isCuda=True):
        self.isCuda=isCuda
        self.O_net = Nets.O_net()

        if self.isCuda:
            self.O_net.cuda()

        self.O_net.load_state_dict(torch.load(O_net_param))

        self.O_net.eval()

        self.m = transforms.ToTensor()

    def detect(self, image, R_net_boxes):
        if len(R_net_boxes)==0:
            return np.array([])

        _img_dataset = []  # 用于存放R网络输出的框对应在原图上的框
        _R_net_boxes = convert_to_square(R_net_boxes)  # 将R网络输出的框转化为正方形

        # 按照新的正方形在原图上抠图
        for _box in _R_net_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            # 将扣的新图传入到O网络中
            img_data = self.m(img)

            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)



        if self.isCuda:
            img_dataset = img_dataset.cuda()
        cond_, position_offset_,landmark_offset_ = self.O_net(img_dataset)

        cond = cond_.cpu().detach()
        position_offset = position_offset_.cpu().detach()
        landmark_offset=landmark_offset_.cpu().detach()

        boxes=[]
        indeces, _ = np.where(cond >=0.99) # _表示占位，返回值舍弃不用
        for index in indeces:
            # 得到R网络输出框的坐标值作为O网络的建议框，以便反算得到O网络输出在原图上的真实框
            _box = _R_net_boxes[index]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * position_offset[index][0]
            y1 = _y1 + oh * position_offset[index][1]
            x2 = _x2 + ow * position_offset[index][2]
            y2 = _y2 + oh * position_offset[index][3]

            fx1 = _x1 + ow * landmark_offset[index][0]
            fy1 = _y1 + oh * landmark_offset[index][1]
            fx2 = _x1 + ow * landmark_offset[index][2]
            fy2 = _y1 + oh * landmark_offset[index][3]
            fx3 = _x1 + ow * landmark_offset[index][4]
            fy3 = _y1 + oh * landmark_offset[index][5]
            fx4 = _x1 + ow * landmark_offset[index][6]
            fy4 = _y1 + oh * landmark_offset[index][7]
            fx5 = _x1 + ow * landmark_offset[index][8]
            fy5 = _y1 + oh * landmark_offset[index][9]
            if abs(x1)<abs(fx3)<abs(x2) and abs(y1)<abs(fy3)<abs(y2):

               boxes.append([x1, y1, x2, y2, cond[index][0],fx1,fy1,fx2,fy2,fx3,fy3,fx4,fy4,fx5,fy5])

        return nms(np.array(boxes), 0.1)