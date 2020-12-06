import PIL.Image as pim
from PIL import ImageDraw
import time
import torch
import P_net_detect, R_net_detect, O_net_detect
import cv2
import numpy as np
from PIL import Image

if __name__ == "__main__":
    with torch.no_grad():
        video_file = r"D:\Pycharm\MTCNN人脸识别加五官\测试样本集\test10.mp4"
        P_detect = P_net_detect.P_net_detector()
        R_detect = R_net_detect.R_net_detector()
        O_detect = O_net_detect.O_net_detector()
        cap = cv2.VideoCapture(video_file)
        # i = 0
        i = 0
        while cap.isOpened():
            sucess, frame = cap.read()  # 按帧读取
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(r"D:\Pycharm\MTCNN人脸识别加五官\测试结果保存\1.avi", fourcc, 20.0,(int(cap.get(3)), int(cap.get(4))))
            # out = cv2.VideoWriter(r"D:\Pycharm\MTCNN人脸识别加五官\测试结果保存\1.mp4", fourcc, fps, (width, height))

            if sucess:
                i = i + 1
                if i % 5 == 0:

                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)

                    P_boxes = P_detect.detect(img)
                    torch.cuda.empty_cache()
                    R_boxes = R_detect.detect(img, P_boxes)
                    torch.cuda.empty_cache()
                    O_boxes = O_detect.detect(img, R_boxes)
                    torch.cuda.empty_cache()

                    for b in O_boxes:  # 把找到的所有人脸框按行赋给b
                        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)  # 画框
                        # cv2.circle(draw, (b[5], b[6]), 1, (0, 0, 255), 2)
                        # cv2.circle(draw, (b[7], b[8]), 1, (0, 0, 255), 2)
                        # cv2.circle(draw, (b[9], b[10]), 1, (0, 0, 255), 2)
                        # cv2.circle(draw, (b[11], b[12]), 1, (0, 0, 255), 2)
                        # cv2.circle(draw, (b[13], b[14]), 1, (0, 0, 255), 2)

                        # cv2.putText(draw, '%f' % float(b[4]), (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

                    out.write(frame)
                    cv2.namedWindow("O_boxes", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("O_boxes",1344,512)
                    cv2.imshow("O_boxes", frame)
                    cv2.waitKey(1)  # 等待多少毫秒
            else:
                break

        # cap.release()
        # out.release()
        # cv2.destroyAllWindows()


