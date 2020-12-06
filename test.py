#
# path=r"C:\MTCNN训练数据样本加五官\48\negative.txt"
#
# file = open(path, "w")
# x=0
# for i in range(301371):
#     x=x+1
#     file.write(
#         "negative\{0}.jpg 0 {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14}\n".format(x, 0, 0,
#                                                                                                    0, 0, 0, 0, 0, 0, 0,
#                                                                                                    0, 0, 0, 0, 0))

# import numpy as np
import PIL.Image as pimg
import os

# imgs = r"C:\hand"
# train_img=r"C:\hhh\48"
# i=300179
# for name in os.listdir(imgs):
#     i=i+1
#     bg = pimg.open(os.path.join(imgs,name))
#     img = bg.convert("RGB")
#     bg_img = img.resize((48,48))
#     bg_img.save("{0}/{1}".format(train_img,str(i)+".jpg"))

# path=r"D:\Pycharm\MTCNN人脸识别加五官\测试样本集\test35.jpg"
# save_path=r"D:\Pycharm\MTCNN人脸识别加五官\测试样本集"
# img=pimg.open(path)
# img_=img.resize((500,188),pimg.ANTIALIAS)
# img_.save(save_path+"/1.jpg")
a=3