import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Fire(nn.module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand1x3_channels):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expend1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expend1x3 = nn.Conv2d(squeeze_channels, expand1x3_channels, kernel_size=(1,3), padding=(0,1))
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)

class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.time_feat = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1,5), padding=(0,2), stride=(1,2)),  #batchsize*1024*60*64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), ceil_mode=True),     #batchsize*1024*30*64
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),                                             #batchsize*1024*30*128
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), ceil_mode=True),     #batchsize*1024*15*128
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),                                           #batchsize*1024*15*256
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), ceil_mode=True),     #batchsize*1024*7*256
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),                                           #batchsize*1024*7*512
        )
        self.Final_avg = nn.AvgPool2d(kernel_size=(1,7))                       #batchsize*1024*1*512
        self.Final_cov = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1) #batchsize*1024*1*2
        self.vfeat_BN_1 = nn.BatchNorm1d(num_features=2048)
        self.vfeat_fc_1 = nn.Linear(in_features=2048, out_features=512)
        self.vfeat_BN_2 = nn.BatchNorm1d(num_features=512)
        self.vfeat_fc_2 = nn.Linear(in_features=512, out_features=512)
        self.afeat_BN_1 = nn.BatchNorm1d(num_features=256)
        self.afeat_fc_1 = nn.Linear(in_features=256, out_features=256)
        self.afeat_BN_2 = nn.BatchNorm1d(num_features=256)
        self.afeat_fc_2 = nn.Linear(in_features=256, out_features=512)
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                if m is self.Final_cov:
                    nn.init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, vfeat, afeat):
        vfeat = vfeat.contiguous()
        vfeat = vfeat.view(vfeat.size(0), 1, vfeat.size(2), vfeat.size(1))
        afeat = afeat.contiguous()
        afeat = afeat.view(afeat.size(0), 1, afeat.size(2), afeat.size(1))
        avfeat = torch.cat([afeat,vfeat], 2)
        avfeat = F.relu(self.Final_cov(self.Final_avg(self.time_feat(avfeat))))#batchsize*1152*1*2
        afeat = avfeat[:, :, 0:127, :]
        afeat = afeat.contiguous()
        afeat = afeat.view(-1, 256)
        vfeat = avfeat[:, :, 128:1152, :]
        vfeat = vfeat.contiguous()
        vfeat = vfeat.view(-1, 2048)
        afeat = nn.ReLU(self.afeat_fc_2(self.afeat_BN_2(nn.ReLU(nn.Dropout(self.afeat_fc_1(self.afeat_BN_1(afeat))), inplace=True))), inplace=True)
        vfeat = nn.ReLU(self.vfeat_fc_2(self.vfeat_BN_2(nn.ReLU(nn.Dropout(self.vfeat_fc_1(self.vfeat_BN_1(vfeat))), inplace=True))), inplace=True)
        return F.pairwise_distance(vfeat, afeat)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        loss = torch.mean((1-label) * torch.pow(dist, 2).squeeze() +
                (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2).squeeze())

        return loss














