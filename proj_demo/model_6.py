import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class FeatAggregate(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, cell_num=2):
        super(FeatAggregate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.rnn = nn.LSTM(input_size, hidden_size, cell_num, batch_first=True)

    def forward(self, feats):
        h0 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)
        c0 = Variable(torch.randn(self.cell_num, feats.size(0), self.hidden_size), requires_grad=False)

        if feats.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        # aggregated feature
        feat, _ = self.rnn(feats, (h0, c0))
        return feat

# Visual-audio multimodal metric learning: LSTM*2+FC*2
class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        self.VFeatPool = FeatAggregate(1024, 128)  #batch* 120 *128
        self.AFeatPool = FeatAggregate(128, 128)   #batch* 120 *128
        #merge batch* 2 * 120 * 128
        self.conv_1 = nn.Conv2d(2, 64, kernel_size=(3, 128), padding=(1,0), stride=(2,1)) #batch*64*60*1
        # transpose batch*1*60*64
        self.mpool_1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1)) # batch*1*30*32
        self.BN_1 = nn.BatchNorm2d(1)
        self.conv_2 = nn.Conv2d(1,32, kernel_size=(3,3),padding=(1,1),stride=(2,1)) # batch*32*15*32
        self.BN_2 = nn.BatchNorm2d(32)
        self.mpool_2 = nn.MaxPool2d(kernel_size=(3,2)) #batch*32*5*16
        self.conv_3 = nn.Conv2d(32,64,kernel_size=(5,3),padding=(0,1),stride=(1,2)) # batch*64*1*8
        # transpose batch*1*64*8
        self.BN_3 = nn.BatchNorm2d(1)
        self.fc_1 = nn.Linear(64*8,1)

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
        vfeat = self.VFeatPool(vfeat)
        afeat = self.AFeatPool(afeat)
        vfeat = vfeat.contiguous()
        vfeat = vfeat.view(vfeat.size(0), 1, vfeat.size(1), vfeat.size(2))     #batch * 1 * 1024 * 120
        afeat = afeat.contiguous()
        afeat = afeat.view(afeat.size(0), 1, afeat.size(1), afeat.size(2))
        avfeat = torch.cat((afeat, vfeat), 1)
        avfeat = self.conv_1(avfeat)
        avfeat = avfeat.transpose(1,3)
        avfeat = F.relu(avfeat, inplace=True)
        avfeat = self.mpool_1(avfeat)
        avfeat = self.BN_1(avfeat)

        avfeat = self.conv_2(avfeat)
        avfeat = F.relu(avfeat, inplace=True)
        avfeat = self.mpool_2(avfeat)
        avfeat = self.BN_2(avfeat)

        avfeat = self.conv_3(avfeat)
        avfeat = F.relu(avfeat)
        avfeat = avfeat.transpose(1,2)
        avfeat = self.BN_3(avfeat)
        avfeat = avfeat.contiguous()
        avfeat = avfeat.view(-1,64*8)
        avfeat = self.fc_1(avfeat)

        return F.sigmoid(avfeat)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, dist, label):
        loss = -torch.mean((1-label) * torch.log(1-dist).squeeze() +
                (label) * torch.log(dist).squeeze())

        return loss


