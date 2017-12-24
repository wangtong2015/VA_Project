import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

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
        X_2 = F.relu(X_2,inplace=True)
        X_2 = self.cov2_2(X_2)
        X_2 = self.BN_2_2(X_2)

        X_3 = self.cov3_1(X_3)
        X_3 = self.BN_3_1(X_3)
        X_3 = F.relu(X_3, inplace=True)
        X_3 = self.cov3_2(X_3)
        X_3 = self.BN_3_2(X_3)

        X_4 = self.maxpool(X_4)
        X_4 = self.cov4_2(X_4)
        X_4 = self.BN_4_2(X_4)

        X = torch.cat((X, X_2, X_3, X_4), 1)
        X = F.relu(X, inplace=True)
        return X

class FeatAggregate(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, cell_num=2):
        super(FeatAggregate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.rnn = nn.LSTM(input_size, hidden_size, cell_num, batch_first=True , dropout=1) #attention::to prevent overfitting

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
        self.VFeatPool = FeatAggregate(1024, 256)  #batch* 120 *256
        self.AFeatPool = FeatAggregate(128, 128)   #batch* 120 *128
        self.cov_1 = nn.Conv2d(3, 32, kernel_size=(5,5), padding=(2, 2), stride=(2,2))  #batch*32*60*64
        self.mpool_0 = nn.MaxPool2d(kernel_size=(3,3), padding=(1,1), stride=(2,2))     #batch*32*30*32
        self.inception_1 = inception(in_channels=32,num_1=32,num_3_1=48,num_3=64,num_5_1=8,num_5=16,num_m=16) #batch*128*30*32
        self.mpool_1 = nn.MaxPool2d(kernel_size=(3,3), padding=(1,1), stride=(2,2))     #batch*128*15*16
        self.inception_2 = inception(in_channels=128,num_1=40,num_3_1=64,num_3=80,num_5_1=10,num_5=20,num_m=20) #batch*160*15*16
        # attention :: since inception block hold the size of the matric ,you can add more inception blocks
        self.mpool_2 = nn.MaxPool2d(kernel_size=(3,3), padding=(1,1), stride=(2,2))     #batch*160*8*8
        self.inception_3 = inception(in_channels=160,num_1=60,num_3_1=80,num_3=120,num_5_1=15,num_5=30,num_m=30) #batch*240*8*8
        self.final_cov = nn.Conv2d(in_channels=240, out_channels=240, kernel_size=(4,4), stride=(4,4))     #batch*240*2*2
        self.BN = nn.BatchNorm2d(240)
        self.fc = nn.Linear(240*4,1)
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
        avfeat = torch.cat((vfeat[:,:,:,0:128],vfeat[:,:,:,128:256],afeat), 1)
        avfeat = self.cov_1(avfeat)
        avfeat = F.relu(avfeat, inplace=True)
        avfeat = self.mpool_0(avfeat)
        avfeat = self.inception_1(avfeat)
        avfeat = self.mpool_1(avfeat)
        avfeat = self.inception_2(avfeat)
        avfeat = self.mpool_2(avfeat)
        avfeat = self.inception_3(avfeat)
        avfeat = self.final_cov(avfeat)
        avfeat = self.BN(avfeat)
        avfeat = avfeat.contiguous()
        avfeat = avfeat.view(-1,240*4)
        avfeat = self.fc(avfeat)
        return F.sigmoid(avfeat)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, dist, label):
        loss = -torch.mean((1-label) * torch.log(1-dist+0.000001).squeeze() +
                (label) * torch.log(dist+0.000001).squeeze())

        return loss



