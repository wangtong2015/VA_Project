import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.vfeat_fc_1 = nn.Conv2d(1, 256, kernel_size=(1024, 1))
        self.vfeat_fc_2 = nn.Conv2d(1, 128, kernel_size=(256, 1))

        self.afeat_fc_1 = nn.Conv2d(1, 128, kernel_size=(128, 1))
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
        vfeat = torch.transpose(vfeat, 1, 2)                             #batch * 1 * 256 * 120
        vfeat = F.relu(vfeat, inplace= True)
        vfeat = self.vfeat_fc_2(vfeat)                                   #batch * 128 * 1  * 120
        vfeat = torch.transpose(vfeat, 1, 2)                             #batch * 1 * 128 * 120
        vfeat = F.relu(vfeat, inplace= True)
        vfeat = vfeat.contiguous()
        vfeat = vfeat.view(-1, 15360)

        afeat = self.afeat_fc_1(afeat)                                   #batch * 128 * 1 * 120
        afeat = torch.transpose(afeat, 1, 2)                             #batch * 1 * 128 * 120
        afeat = F.relu(afeat, inplace= True)
        afeat = afeat.contiguous()
        afeat = afeat.view(-1, 15360)
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
