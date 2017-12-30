import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import model_6_3

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

class VAMetric(nn.Module):
    def __init__(self):
        super(VAMetric, self).__init__()
        pretrained = model_6_3.VAMetric()
        pretrained.load_state_dict(torch.load("./checkpoints/VA_METRIC_state_epoch40.pth"))
        self.conv_1 = nn.Conv2d(3,64,(3,128),(2,1),(1,0))         #batch* 64* 60 *1
        self.BN_1 = nn.BatchNorm2d(1)
        self.conv_2 = nn.Conv2d(1, 32, (5,5), 2, 2)               #batch*32*30*32
        self.BN_2 = nn.BatchNorm2d(32)
        self.conv_3_1 = nn.Conv2d(32, 32, (3,3), (2,1), (1,1))
        self.conv_3_2 = nn.Conv2d(32, 32, (3,5), (2,1), (1,2))    #batch*64*15*32
        self.mpool_3 = nn.MaxPool2d(kernel_size=2, padding=(1,0)) #batch*64*8*16
        self.BN_3 = nn.BatchNorm2d(64)
        self.inctption_1 = inception(64, 32, 64, 48, 16, 12, 16)  #batch*128*8*16
        self.apool_1 = nn.AvgPool2d(2)                             #batch*128*4*8
        self.conv_4 = nn.Conv2d(128, 256, (2,2),2)                #batch*256*2*4
        self.apool_2 = nn.AvgPool2d((2,1))                        #batch*256*1*4
        self.fc = nn.Linear(1024,1)
        self.init_params()
        self.VFeatPool = pretrained.VFeatPool                     #batch* 120 *128
        self.AFeatPool = pretrained.AFeatPool                     #batch* 120 *128
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
        avfeat = self.conv_1(avfeat)
        avfeat = avfeat.transpose(1,3)
        avfeat = F.relu(avfeat, inplace=True)
        avfeat = self.BN_1(avfeat)
        avfeat = self.conv_2(avfeat)
        avfeat = F.relu(avfeat, inplace=True)
        avfeat = self.BN_2(avfeat)
        avfeat_1 = self.conv_3_1(avfeat)
        avfeat_2 = self.conv_3_2(avfeat)
        avfeat = torch.cat((avfeat_1, avfeat_2), 1)
        avfeat = F.relu(avfeat, inplace=True)
        avfeat = self.mpool_3(avfeat)
        avfeat = self.BN_3(avfeat)
        avfeat = self.inctption_1(avfeat)
        avfeat = self.apool_1(avfeat)
        avfeat = self.conv_4(avfeat)
        avfeat = F.relu(avfeat,inplace=True)
        avfeat = self.apool_2(avfeat)
        avfeat = avfeat.contiguous()
        avfeat = avfeat.view(-1,1024)
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
        loss = -torch.mean((1-label) * torch.log(1-dist+0.0001).squeeze() +
                (label) * torch.log(dist+0.0001).squeeze())

        return loss










