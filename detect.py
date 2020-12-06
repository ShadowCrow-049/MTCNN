import PIL.Image as pimg
from PIL import ImageDraw
import time
import torch
import P_net_detect,R_net_detect,O_net_detect

if __name__=="__main__":
  # with torch.no_grad():
    image_file = r"测试样本集\7c7acd6a1af2969dd97cf477700ce33.jpg"
    P_detect=P_net_detect.P_net_detector()
    R_detect=R_net_detect.R_net_detector()
    O_detect=O_net_detect.O_net_detector()

    with pimg.open(image_file) as img:
     torch.cuda.empty_cache()
     with torch.no_grad():
        img = img.convert("RGB")
        starttime=time.time()
        P_boxes=P_detect.detect(img)
        torch.cuda.empty_cache()
        endtime=time.time()
        P_time=endtime-starttime

        starttime=time.time()
        R_boxes=R_detect.detect(img,P_boxes)
        torch.cuda.empty_cache()
        endtime=time.time()
        R_time=endtime-starttime

        starttime=time.time()
        O_boxes=O_detect.detect(img,R_boxes)
        torch.cuda.empty_cache()
        endtime=time.time()
        O_time=endtime-starttime

        sumtime = P_time + R_time + O_time
        print("总时间：{0} P时间：{1} R时间：{2} O时间：{3}".format(sumtime, P_time, R_time, O_time))


        imgdraw = ImageDraw.Draw(img)
        for box in O_boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            fx1=int(box[5])
            fy1 = int(box[6])
            fx2 = int(box[7])
            fy2 = int(box[8])
            fx3 = int(box[9])
            fy3 = int(box[10])
            fx4 = int(box[11])
            fy4 = int(box[12])
            fx5 = int(box[13])
            fy5 = int(box[14])

            print(box[4])

            imgdraw.rectangle((x1, y1, x2, y2), fill=None, outline="red",width=1)

            # imgdraw.point((fx1,fy1),"#FFFF00")
            # imgdraw.point((fx2, fy2), "#FFFF00")
            # imgdraw.point((fx3, fy3), "#FFFF00")
            # imgdraw.point((fx4, fy4), "#FFFF00")
            # imgdraw.point((fx5, fy5), "#FFFF00")
            i=1
            imgdraw.rectangle((fx1, fy1, fx1 + i, fy1 + i), fill="#FFFF00", outline="#FFFF00")
            imgdraw.rectangle((fx2, fy2, fx2 + i, fy2 + i), fill="#FFFF00", outline="#FFFF00")
            imgdraw.rectangle((fx3, fy3, fx3 + i, fy3 + i), fill="#FFFF00", outline="#FFFF00")
            imgdraw.rectangle((fx4, fy4, fx4 + i, fy4 + i), fill="#FFFF00", outline="#FFFF00")
            imgdraw.rectangle((fx5, fy5, fx5 + i, fy5 + i), fill="#FFFF00", outline="#FFFF00")

        img.show()




