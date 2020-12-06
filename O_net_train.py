from Nets import O_net
import trainer
if __name__=='__main__':
    net=O_net()
    trainer=trainer.Trainer(net,r"D:\Pycharm\MTCNN人脸识别加五官\网络参数\O_net.pt",r"C:\MTCNN训练数据样本加五官\48")
    trainer.train()