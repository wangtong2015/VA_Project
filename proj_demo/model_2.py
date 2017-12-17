import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class inception(nn.Module):            # a inception block
    def __init__(self, in_channels, num_1, num_3, num_3_1, num_5, num_5_1, num_m):
        super(inception, self).__init__()
        self.cov1_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_1, kernel_size=(1,1))
        self.cov2_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_3_1, kernel_size=(1,1))
        self.cov3_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_5_1, kernel_size=(1,1))
        self.BN_1_1 = nn.BatchNorm2d(num_features=num_1)
        self.BN_2_1 = nn.BatchNorm2d(num_features=num_3_1)
        self.BN_2_2 = nn.BatchNorm2d(num_features=num_3)
        self.BN_3_1 = nn.BatchNorm2d(num_features=num_5_1)
        self.BN_3_2 = nn.BatchNorm2d(num_features=num_5)
        self.BN_4_2 = nn.BatchNorm2d(num_features=num_m)
        self.cov4_2 = nn.Conv2d(in_channels=in_channels, out_channels=num_m, kernel_size=(1,1))
        self.cov2_2 = nn.Conv2d(in_channels=num_3_1, out_channels=num_3, kernel_size=(3,3), padding=(1,1))
        self.cov3_2 = nn.Conv2d(in_channels=num_5_1, out_channels=num_5, kernel_size=(5,5), padding=(2,2))
        self.maxpool = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1))
    def forward(self, X):
        X_2 = X
        X_3 = X
        X_4 = X
        X = self.cov1_1(X)
        X = self.BN_1_1(X)

        X_2 = self.cov2_1(X_2)
        X_2 = self.BN_2_1(X_2)
        X_2 = F.relu(X_2)
        X_2 = self.cov2_2(X_2)
        X_2 = self.BN_2_2(X_2)

        X_3 = self.cov3_1(X_3)
        X_3 = self.BN_3_1(X_3)
        X_3 = F.relu(X_3)
        X_3 = self.cov3_2(X_3)
        X_3 = self.BN_3_2(X_3)

        X_4 = self.maxpool(X_4)
        X_4 = self.cov4_2(X_4)
        X_4 = self.BN_4_2(X_4)

        X = torch.cat((X, X_2, X_3, X_4), 1)
        X = F.relu(X)
        return X
class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.v_cov_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,7), padding=(3,3), stride=(2,2))  #batch*64*512*60
        self.v_BN_1 = nn.BatchNorm2d(num_features=64)
        self.v_maxpool_1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))            #batch*64*256*30
        self.v_cov_2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3,3), padding=(1,1),stride=(2,2))#batch*192*128*15
        self.v_BN_2 = nn.BatchNorm2d(num_features=192)
        self.v_maxpool_2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))             #batch*192*64*8
        self.v_inception = inception(in_channels=192, num_1=64, num_3=128, num_3_1=96, num_5= 32, num_5_1=16, num_m=32)  #batch * 256 * 128* 8
        self.v_avgpool = nn.AvgPool2d(kernel_size=(64,8))        #batch*256*1*1
        self.v_fc = nn.Linear(256,256)
        self.v_BN_3 = nn.BatchNorm1d(256)

        self.a_cov_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,7), padding=(3,3), stride=(2,2))  #batch*64*64*60
        self.a_BN_1 = nn.BatchNorm2d(num_features=64)
        self.a_maxpool_1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))            #batch*64*32*30
        self.a_cov_2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3,3), padding=(1,1),stride=(2,2))#batch*192*16*15
        self.a_BN_2 = nn.BatchNorm2d(num_features=192)
        self.a_maxpool_2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))             #batch*192*8*8
        self.a_inception = inception(in_channels=192, num_1=64, num_3=128, num_3_1=96, num_5= 32, num_5_1=16, num_m=32)  #batch * 256 * 8* 8
        self.a_avgpool = nn.AvgPool2d(kernel_size=(8,8))        #batch*256*1*1
        self.a_fc = nn.Linear(256,256)
        self.a_BN_3 = nn.BatchNorm1d(256)
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
    def forward(self, vfeat, afeat):
        vfeat = vfeat.contiguous()
        vfeat = vfeat.view(vfeat.size(0), 1, vfeat.size(2), vfeat.size(1))
        vfeat = self.v_cov_1(vfeat)
        vfeat = self.v_BN_1(vfeat)
        vfeat = F.relu(vfeat)
        vfeat = self.v_maxpool_1(vfeat)
        vfeat = self.v_cov_2(vfeat)
        vfeat = self.v_BN_2(vfeat)
        vfeat = F.relu(vfeat)
        vfeat = self.v_maxpool_2(vfeat)
        vfeat = self.v_inception(vfeat)
        vfeat = self.v_avgpool(vfeat)
        vfeat = vfeat.contiguous()
        vfeat = vfeat.view(-1,256)
        vfeat = self.v_BN_3(vfeat)
        vfeat = self.v_fc(vfeat)

        afeat = afeat.contiguous()
        afeat = afeat.view(afeat.size(0), 1, afeat.size(2), afeat.size(1))
        afeat = self.a_cov_1(afeat)
        afeat = self.a_BN_1(afeat)
        afeat = F.relu(afeat)
        afeat = self.a_maxpool_1(afeat)
        afeat = self.a_cov_2(afeat)
        afeat = self.a_BN_2(afeat)
        afeat = F.relu(afeat)
        afeat = self.a_maxpool_2(afeat)
        afeat = self.a_inception(afeat)
        afeat = self.a_avgpool(afeat)
        afeat = afeat.contiguous()
        afeat = afeat.view(-1,256)
        afeat = self.a_BN_3(afeat)
        afeat = self.a_fc(afeat)

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
