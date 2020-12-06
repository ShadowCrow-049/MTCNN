import torch.nn as nn

class P_net(nn.Module):
    def __init__(self):
        super(P_net,self).__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=10),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            # N*5*5*10
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            # N*3*3*16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
            # N*1*1*32
        )
        # N*1*1*1
        self.last_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1), stride=1, padding=0),
            # nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
            # N*1*1*1
        )
        self.last_conv2 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=4, kernel_size=(1, 1), stride=1, padding=0),
        # nn.BatchNorm2d(num_features=4),
        # N*1*1*4
       )

        self.last_conv3 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), stride=1, padding=0),
        # nn.BatchNorm2d(num_features=4),
        # N*1*1*10
    )
    def forward(self, x):

        y=self.pre_conv(x)
        cond=self.last_conv1(y)
        position_offset=self.last_conv2(y)
        landmark_offset=self.last_conv3(y)

        return cond,position_offset,landmark_offset



class R_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=28),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            # N*11*11*28
            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # N*4*4*48
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(2, 2), stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
            # N*3*3*64
         )
        self.pre_line=nn.Sequential(
            nn.Linear(in_features=3 * 3 * 64, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.last_conv1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
        self.last_conv2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=4)
        )
        self.last_conv3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10)
        )
    def forward(self, x):
        y=self.pre_conv(x)
        y = y.reshape(-1,3*3*64)
        y1=self.pre_line(y)
        cond=self.last_conv1(y1)
        position_offset=self.last_conv2(y1)
        landmark_offset=self.last_conv3(y1)

        return cond,position_offset,landmark_offset


class O_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            # N*23*23*32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # N*10*10*64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # N*4*4*64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=1, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
           # N*3*3*128
        )
        self.pre_line = nn.Sequential(
            nn.Linear(in_features=3 * 3 * 128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.last_conv1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )
        self.last_conv2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=4)
        )
        self.last_conv3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=10)
        )
    def forward(self, x):
        y=self.pre_conv(x)
        y=y.reshape(-1,3*3*128)
        y1=self.pre_line(y)
        cond=self.last_conv1(y1)
        position_offset=self.last_conv2(y1)
        landmark_offset=self.last_conv3(y1)

        return cond,position_offset,landmark_offset
