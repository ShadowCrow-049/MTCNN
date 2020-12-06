import numpy as np
import os
from PIL import  ImageFilter
import PIL.Image as pimg
from tools import iou


img_path = r"D:\Pycharm\MTCNN人脸识别\img_celeba"
label_position_path = r'D:\Pycharm\MTCNN人脸识别加五官\list_bbox_celeba.txt'
label_landmark_path = r"D:\Pycharm\MTCNN人脸识别加五官\list_landmarks_celeba.txt"
handel_path = r"C:\MTCNN训练数据样本加五官"


#创建用于存放相应尺寸训练样本的文件夹，里面分别有对应的正样本、负样本和部分样本
def mkdir(size):
    rootpath = os.path.join(handel_path, str(size))
    if not os.path.exists(rootpath):
        os.mkdir(rootpath)

    p_dirpath = os.path.join(rootpath, "positive")
    if not os.path.exists(p_dirpath):
        os.mkdir(p_dirpath)

    n_dirpath = os.path.join(rootpath, "negative")
    if not os.path.exists(n_dirpath):
        os.mkdir(n_dirpath)

    t_dirpath = os.path.join(rootpath, "part")
    if not os.path.exists(t_dirpath):
        os.mkdir(t_dirpath)

    return rootpath, p_dirpath, n_dirpath, t_dirpath


def sample_handle(size):
    positive_count = 0   # 用于计数，记录生成了多少张图片
    negative_count=0
    part_count=0
    r_path, p_path, n_path, t_path = mkdir(size)  # 创建目录
    # 生成保存标签的文件
    p_file = open(r_path + "/positive.txt", "w")
    n_file = open(r_path + "/negative.txt", "w")
    t_file = open(r_path + "/part.txt", "w")

    f_position = open(label_position_path).readlines()
    f_landmark = open(label_landmark_path).readlines()

    for index in range(len(f_landmark)):     # 读取标签中的每一行
        if index < 2:    # 跳过前两行
            continue

        strs_postion = f_position[index].strip().split(" ")     # 去除空格并进行切分以获取位置信息
        strs_landmark = f_landmark[index].strip().split(" ")
        strs_postion = list(filter(bool, strs_postion))             # 对不需要的信息进行过滤
        strs_landmark = list(filter(bool, strs_landmark))

        filename = strs_postion[0]    # 获取文件名
        # 获取原始坐标
        x1 = float(strs_postion[1])
        y1 = float(strs_postion[2])
        w = float(strs_postion[3])
        h = float(strs_postion[4])
        x2 = x1 + w
        y2=y1+h

        # 5个特征点的位置
        fx1 = float(strs_landmark[1])
        fy1 = float(strs_landmark[2])
        fx2 = float(strs_landmark[3])
        fy2 = float(strs_landmark[4])
        fx3 = float(strs_landmark[5])
        fy3 = float(strs_landmark[6])
        fx4 = float(strs_landmark[7])
        fy4 = float(strs_landmark[8])
        fx5 = float(strs_landmark[9])
        fy5 = float(strs_landmark[10])

        # 去除奇异样本
        if max(w, h) < 40 or x1 < 0 or y1 < 0 or w <= 5 or h <= 5:
            continue
        # 计算中心点
        cx = x1 + w * 0.5
        cy = y1 + h * 0.5
        img = pimg.open(os.path.join(img_path,filename))     # 读取对应的图片
        width, high = img.size   # 获取原始图片的宽和高方便以后生成负样本

        # 循环生成训练样本
        for count in range(5):
            # 随机浮动产生正方形正、负、部分样本
            # 设定中心点的偏移量

            offset_x = np.random.randint(-w*0.2, w*0.2)
            offset_y = np.random.randint(-h*0.2, h*0.2)
            # # 计算偏移后的中心点
            _cx = cx + offset_x
            _cy = cy + offset_y

            # 设定偏移后的边界框宽度
            _side=np.random.randint(int(min(w,h)*0.5),np.ceil(1.25*max(w,h)))

            # 偏移后的起始坐标(要防止越界)
            _x1 = np.maximum(_cx - _side * 0.5, 0)
            _y1 = np.maximum(_cy - _side * 0.5, 0)
            _x2 = _x1 + _side
            _y2 = _y1 + _side

            # 计算建议框偏移值
            offset_x1 = (x1 - _x1) / _side
            offset_y1 = (y1 - _y1) / _side
            offset_x2 = (x2 - _x2) / _side
            offset_y2 = (y2 - _y2) / _side

            # 计算五个关键点偏移值
            offset_fx1 = (fx1 - _x1) / _side
            offset_fy1 = (fy1 - _y1) / _side
            offset_fx2 = (fx2 - _x1) / _side
            offset_fy2 = (fy2 - _y1) / _side
            offset_fx3 = (fx3 - _x1) / _side
            offset_fy3 = (fy3 - _y1) / _side
            offset_fx4 = (fx4 - _x1) / _side
            offset_fy4 = (fy4 - _y1) / _side
            offset_fx5 = (fx5 - _x1) / _side
            offset_fy5 = (fy5 - _y1) / _side

            # 计算IOU
            # [x1, y1, x2, y2, 置信度]
            box = np.array([x1, y1, x2, y2, 0])
            boxs = np.array([[_x1, _y1, _x2, _y2, 0]])
            per = iou(box, boxs, False)
            per = per[0]   # 获取IOU的数值

            tempimg = img.crop((_x1, _y1, _x2, _y2))      # 截取图片
            tempimg = tempimg.resize((size, size), pimg.ANTIALIAS)   # 将裁剪出的图片缩放成对应的大小并设置为抗锯齿防止因缩放造成过多的信息丢失
            # 创建一个列表用于保存原始图片和经过模糊处理后的图片
            imglist = []
            imglist.append(tempimg)

            filterimg = tempimg.filter(ImageFilter.BLUR)    # 图片模糊处理
            imglist.append(filterimg)
            for _tempimg in imglist:
                if per > 0.65:  # 正样本
                    positive_count += 1
                    _tempimg.save("{0}/{1}.jpg".format(p_path, positive_count))
                    p_file.write(
                        "positive\{0}.jpg 1 {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14}\n".format(positive_count,offset_x1,offset_y1,offset_x2,offset_y2,offset_fx1,offset_fy1,offset_fx2,offset_fy2,offset_fx3,offset_fy3,offset_fx4,offset_fy4,offset_fx5,offset_fy5))

                elif per < 0.1:  # 负样本
                    negative_count += 1
                    _tempimg.save("{0}/{1}.jpg".format(n_path, negative_count))
                    n_file.write("negative\{0}.jpg 0 {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14}\n".format(negative_count, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

                elif (per > 0.2) and (per < 0.4):  # 部分样本
                    part_count += 1
                    _tempimg.save("{0}/{1}.jpg".format(t_path, part_count))
                    t_file.write(
                        "part\{0}.jpg 2 {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14}\n".format(part_count,offset_x1,offset_y1,offset_x2,offset_y2, offset_fx1,offset_fy1,offset_fx2,offset_fy2,offset_fx3,offset_fy3,offset_fx4,offset_fy4,offset_fx5,offset_fy5))

        # 再创建负样本
        for i in range(2):
            _side=np.random.randint(size*0.8,min(width,high)*0.5)
            _x1 = np.random.uniform(0, width - _side)
            _y1 = np.random.uniform(0, high - _side)
            _x2 = _x1 + _side
            _y2 = _y1 + _side

            # 计算IOU
            # [x1, y1, x2, y2, 置信度]
            box = np.array([x1, y1, x2, y2, 0])
            boxs = np.array([[_x1, _y1, _x2, _y2, 0]])
            per = iou(box, boxs, False)
            per = per[0]
            # 截取图片
            tempimg = img.crop((_x1, _y1, _x2, _y2))
            tempimg = tempimg.resize((size, size), pimg.ANTIALIAS)
            imglist = []
            imglist.append(tempimg)
            # filterimg = tempimg.filter(ImageFilter.BLUR)
            # imglist.append(filterimg)
            for _tempimg in imglist:
                if per < 0.1:
                    negative_count += 1
                    _tempimg.save("{0}/{1}.jpg".format(n_path, negative_count))
                    n_file.write("negative\{0}.jpg 0 {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14}\n".format(negative_count, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    p_file.close()
    n_file.close()
    t_file.close()


if __name__ == '__main__':
     sample_handle(12)
    # sample_handle(24)
    #  sample_handle(48)