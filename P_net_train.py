from Nets import P_net
import trainer
if __name__=='__main__':
    net=P_net()
    trainer=trainer.Trainer(net,r"D:\Pycharm\MTCNN人脸识别加五官\网络参数\P_net.pt",r"C:\MTCNN训练数据样本加五官\12")
    trainer.train()


