import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.vfeat_avgpool = nn.AvgPool2d(kernel_size=(1,120))
        self.afeat_avgpool = nn.AvgPool2d(kernel_size=(1,120))
        self.vfeat_fc = nn.Linear(1024,256)
        self.afeat_fc = nn.Linear(128, 64)
        self.avfeat_fc_1 = nn.Linear(320,128)
        self.avfeat_fc_2 = nn.Linear(128,32)
        self.avfeat_fc_3 = nn.Linear(32,1)
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
    def forward(self, vfeat, afeat):
        vfeat = vfeat.contiguous()
        vfeat = vfeat.view(vfeat.size(0), 1, vfeat.size(2), vfeat.size(1))     #batch * 1 * 1024 * 120
        afeat = afeat.contiguous()
        afeat = afeat.view(afeat.size(0), 1, afeat.size(2), afeat.size(1))
        vfeat = self.vfeat_avgpool(vfeat)                                      #batch * 1 * 1024 * 1
        afeat = self.afeat_avgpool(afeat)
        vfeat = vfeat.contiguous()
        vfeat = vfeat.view(-1, 1024)
        afeat = afeat.contiguous()
        afeat = afeat.view(-1, 128 )
        vfeat = self.vfeat_fc(vfeat)
        vfeat = F.relu(vfeat,inplace=True)
        afeat = self.afeat_fc(afeat)
        afeat = F.relu(afeat,inplace=True)
        avfeat = torch.cat((vfeat, afeat), 1)
        avfeat = self.avfeat_fc_1(avfeat)
        avfeat = F.relu(avfeat,inplace=True)
        avfeat = self.avfeat_fc_2(avfeat)
        avfeat = F.relu(avfeat,inplace= True)
        avfeat = self.avfeat_fc_3(avfeat)
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
