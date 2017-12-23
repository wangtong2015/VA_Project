import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#this model have abuout 0.3million parameters totally, only use FC
class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.training = True
        self.vfeat_fc_1 = nn.Conv2d(1, 256, kernel_size=(1024, 1))
        self.vfeat_fc_2 = nn.Conv2d(1, 64, kernel_size=(256, 1))
        self.vfeat_fc_3 = nn.Conv2d(1, 16, kernel_size=(64,1))
        self.vfeat_fc_4 = nn.Conv2d(1, 4, kernel_size=(16,1))

        self.afeat_fc_1 = nn.Conv2d(1, 64, kernel_size=(128, 1))
        self.afeat_fc_2 = nn.Conv2d(1, 32, kernel_size=(64, 1))
        self.afeat_fc_3 = nn.Conv2d(1, 12, kernel_size=(32, 1))
        self.afeat_fc_4 = nn.Conv2d(1, 4, kernel_size=(12, 1))
        #attention::now no dropout do check overfitting
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.Final_cov:
                    nn.init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, vfeat, afeat):
        vfeat = vfeat.contiguous()
        vfeat = vfeat.view(vfeat.size(0), 1, vfeat.size(2), vfeat.size(1))     #batch * 1 * 1024 * 120
        afeat = afeat.contiguous()
        afeat = afeat.view(afeat.size(0), 1, afeat.size(2), afeat.size(1))
        vfeat = self.vfeat_fc_1(vfeat)                                   #batch * 256 * 1 * 120
        vfeat = F.dropout(vfeat, training=self.training)
        vfeat = torch.transpose(vfeat, 1, 2)                             #batch * 1 * 256 * 120
        vfeat = F.relu(vfeat, inplace= True)
        vfeat = self.vfeat_fc_2(vfeat)                                   #batch * 64 * 1  * 120
        vfeat = F.dropout(vfeat,p=0.65, training=self.training)
        vfeat = torch.transpose(vfeat, 1, 2)                             #batch * 1 * 64 * 120
        vfeat = F.relu(vfeat, inplace= True)
        vfeat = self.vfeat_fc_3(vfeat)                                   #batch * 16* 1 * 120
        vfeat = torch.transpose(vfeat, 1, 2)                             #batch * 1 * 16 * 120
        vfeat = F.relu(vfeat, inplace= True)
        vfeat = self.vfeat_fc_4(vfeat)                                   #batch * 4 * 1  * 120
        vfeat = torch.transpose(vfeat, 1, 2)                             #batch * 1 * 4  * 120
        vfeat = F.relu(vfeat, inplace= True)
        vfeat = vfeat.contiguous()
        vfeat = vfeat.view(-1, 480)

        afeat = self.afeat_fc_1(afeat)                                   #batch * 64 * 1 * 120
        afeat = F.dropout(afeat, training=self.training)
        afeat = torch.transpose(afeat, 1, 2)                             #batch * 1 * 64 * 120
        afeat = F.relu(afeat, inplace= True)
        afeat = self.afeat_fc_2(afeat)                                   #batch * 32 * 1  * 120
        afeat = F.dropout(afeat, training=self.training, p=0.65)
        afeat = torch.transpose(afeat, 1, 2)                             #batch * 1 * 32 * 120
        afeat = F.relu(afeat, inplace= True)
        afeat = self.afeat_fc_3(afeat)                                   #batch * 12* 1 * 120
        afeat = torch.transpose(afeat, 1, 2)                             #batch * 1 * 12 * 120
        afeat = F.relu(afeat, inplace= True)
        afeat = self.afeat_fc_4(afeat)                                   #batch * 4 * 1  * 120
        afeat = torch.transpose(afeat, 1, 2)                             #batch * 1 * 4  * 120
        afeat = F.relu(afeat, inplace= True)
        afeat = afeat.contiguous()
        afeat = afeat.view(-1, 480)

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




