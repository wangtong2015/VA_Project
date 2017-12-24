import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.v_cov_1 = nn.Conv2d(1, 16, kernel_size=(17, 1), padding=(8,0), stride=(8,1))
        self.a_cov_1 = nn.Conv2d(1, 4, kernel_size=(5, 1), padding=(2,0))                   #batch* 128 *120 * 20
        self.av_BN_1 = nn.BatchNorm2d(20)
        self.av_cov_1 = nn.Conv2d(20, 40, kernel_size=(9, 3), padding=(4,1), stride=(2, 2))  #batch* 64 * 60 *40
        self.av_BN_1 = nn.BatchNorm2d(40)
        self.av_mpool_1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True, stride=2)               #batch*32*30*40
        self.av_cov_2 = nn.Conv2d(40, 60, kernel_size=(7,3), stride=(2,2))                    #batch*13*14*60
        self.av_BN_2 = nn.BatchNorm2d(60)
        self.av_mpool_2 = nn.MaxPool2d(kernel_size=3, stride=2,padding=(0,1))                 #batch*6* 7*60
        self.av_cov_3 = nn.Conv2d(60, 80, kernel_size=(6,1))                                  #batch*1*7*80
        self.av_apool_3 = nn.AvgPool2d(kernel_size=(1,7))                                     #batch*1*1*80
        self.av_BN_3 = nn.BatchNorm2d(80)
        self.fc = nn.Linear(80,1)
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight.data, mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
    def forward(self, vfeat, afeat):
        vfeat = vfeat.contiguous()
        vfeat = vfeat.view(vfeat.size(0), 1, vfeat.size(2), vfeat.size(1))     #batch * 1 * 1024 * 120
        afeat = afeat.contiguous()
        afeat = afeat.view(afeat.size(0), 1, afeat.size(2), afeat.size(1))
        vfeat = self.v_cov_1(vfeat)
        afeat = self.a_cov_1(afeat)
        avfeat = torch.cat((afeat,vfeat), 1)
        avfeat = F.relu(avfeat, inplace=True)

        avfeat = self.av_cov_1(avfeat)
        avfeat = F.relu(avfeat)
        avfeat = self.av_BN_1(avfeat)
        avfeat = self.av_mpool_1(avfeat)
        avfeat = self.av_cov_2(avfeat)
        avfeat = F.relu(avfeat, inplace=True)
        avfeat = self.av_BN_2(avfeat)
        avfeat = self.av_mpool_2(avfeat)
        avfeat = self.av_cov_3(avfeat)
        avfeat = F.relu(avfeat)
        avfeat = self.av_BN_3(avfeat)
        avfeat = self.av_apool_3(avfeat)
        avfeat = avfeat.contiguous()
        avfeat = avfeat.view(-1,80)
        avfeat = self.fc(avfeat)
        return F.sigmoid(avfeat)

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

