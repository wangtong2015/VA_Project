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
        self.v_cov_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,7), padding=(3,3)) #batch*64*1024*120
        self.v_BN_1 = nn.BatchNorm2d(num_features=64)
        self.v_maxpool_1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))            #batch*64*512*60
        self.v_cov_2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3,3), padding=(1,1))#batch*192*512*60
        self.v_BN_2 = nn.BatchNorm2d(num_features=192)
        self.v_maxpool_2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))             #batch*192*256*30
        self.v_inception_3_a = inception(in_channels=192, num_1=64, num_3=128, num_3_1=96, num_5= 32, num_5_1=16, num_m=32)  #batch * 256 * 256* 30
        self.v_inception_3_b = inception(in_channels=256, num_1=128, num_3=192, num_3_1=128, num_5= 96, num_5_1=32, num_m=64) #batch * 480 * 256 * 30
        self.v_maxpool_3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))      #attention:: repeat model   batch * 480 * 128 * 15
        self.v_inception_4_a = inception(in_channels=480, num_1=192, num_3=208, num_3_1=96, num_5=48, num_5_1=16, num_m=64) #batch * 512 * 128 * 15
        self.v_inception_4_b = inception(in_channels=512, num_1=160, num_3_1=112, num_3=224, num_5_1=24, num_5=64, num_m=64) #batch * 512 * 128 * 15
        self.v_inception_4_c = inception(in_channels=512, num_1=128, num_3_1=128, num_3=256, num_5_1=24, num_5=64, num_m=64) #batch * 512 * 128 * 15
        self.v_inception_4_d = inception(in_channels=512, num_1=112, num_3_1=144, num_3=288, num_5_1=32, num_5=64, num_m=64) #batch * 528 * 128 * 15
        self.v_inception_4_e = inception(in_channels=528, num_1=256, num_3_1=160, num_3=320, num_5_1=32, num_5=128, num_m=128) #batch * 832 * 128 *15
        self.v_maxpool_4 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))      # batch * 832 * 64 * 8
        self.v_inception_5_a = inception(in_channels=832, num_1=256, num_3_1=160, num_3=320, num_5_1=32, num_5=128, num_m=128) #batch * 832 * 64 * 8
        self.v_inception_5_b = inception(in_channels=832, num_1=384, num_3_1=192, num_3=384, num_5_1=48, num_5=128, num_m=128) #batch * 1024 * 64 * 8
        self.v_avgpool_5 = nn.AvgPool2d(kernel_size=(64,8))      #batch * 1024 * 1 * 1
        self.v_FC = nn.Linear(in_features=1024, out_features=512)
        self.v_BN_3 = nn.BatchNorm1d(1024)
        ## add some short cut at 4_a and 4_d stage
        self.v_shortcut_1_cov1 = nn.Conv2d(in_channels=512, kernel_size=(1,1), out_channels=1024)  #batch * 1024 * 128 * 15
        self.v_shortcut_1_avgpool = nn.AvgPool2d(kernel_size=(128,15))   #batch * 1024 * 128 * 15
        self.v_shortcut_2_cov1 = nn.Conv2d(in_channels=528, kernel_size=(1,1), out_channels=1024)  #batch * 1024 * 128 * 15
        self.v_shortcut_2_avgpool = nn.AvgPool2d(kernel_size=(128,15))   #batch * 528 * 1 * 1

        self.a_cov_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7,7), padding=(3,3)) #batch*32*128*120
        self.a_BN_1 = nn.BatchNorm2d(num_features=32)
        self.a_maxpool_1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))            #batch*32*64*60
        self.a_cov_2 = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(3,3), padding=(1,1))#batch*96*64*60
        self.a_BN_2 = nn.BatchNorm2d(num_features=96)
        self.a_maxpool_2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))             #batch*96*32*30
        self.a_inception_3_a = inception(in_channels=96, num_1=32, num_3=64, num_3_1=48, num_5= 16, num_5_1=8, num_m=16)  #batch *128* 32 * 30
        self.a_inception_3_b = inception(in_channels=128, num_1=64, num_3=96, num_3_1=64, num_5= 48, num_5_1=16, num_m=32) #batch * 240*32*30
        self.a_maxpool_3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))      #attention:: repeat model   batch * 240 * 16 * 15
        self.a_inception_4_a = inception(in_channels=240, num_1=96, num_3=104, num_3_1=48, num_5=24, num_5_1=8, num_m=32) #batch * 256 * 16 * 15
        self.a_inception_4_b = inception(in_channels=256, num_1=80, num_3_1=56, num_3=112, num_5_1=12, num_5=32, num_m=32) #batch * 256 * 16 * 15
        self.a_inception_4_c = inception(in_channels=256, num_1=64, num_3_1=64, num_3=128, num_5_1=12, num_5=32, num_m=32) #batch * 256 * 16 * 15
        self.a_inception_4_d = inception(in_channels=256, num_1=56, num_3_1=72, num_3=144, num_5_1=16, num_5=32, num_m=32) #batch * 264 * 16 * 15
        self.a_inception_4_e = inception(in_channels=264, num_1=128, num_3_1=80, num_3=160, num_5_1=16, num_5=64, num_m=64) #batch * 416 * 16 *15
        self.a_maxpool_4 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))      # batch * 416 * 8 * 8
        self.a_inception_5_a = inception(in_channels=416, num_1=128, num_3_1=80, num_3=160, num_5_1=16, num_5=64, num_m=64) #batch * 416 * 8 * 8
        self.a_inception_5_b = inception(in_channels=416, num_1=192, num_3_1=96, num_3=192, num_5_1=24, num_5=64, num_m=64) #batch * 512 * 8 * 8
        self.a_avgpool_5 = nn.AvgPool2d(kernel_size=(8,8))      #batch * 512 * 2 * 2
        self.a_FC = nn.Linear(in_features=512, out_features=512)
        self.a_BN_3 = nn.BatchNorm1d(512)
        ## add some short cut at 4_a and 4_d stage
        self.a_shortcut_1_cov1 = nn.Conv2d(in_channels=256, kernel_size=(1,1), out_channels=512)  #batch * 512 * 16 * 15
        self.a_shortcut_1_avgpool = nn.AvgPool2d(kernel_size=(16,15))                             #batch * 512 * 1 * 1
        self.a_shortcut_2_cov1 = nn.Conv2d(in_channels=264, kernel_size=(1,1), out_channels=512)
        self.a_shortcut_2_avgpool = nn.AvgPool2d(kernel_size=(16,15))                             #batch * 512 * 1 * 1
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
        vfeat = self.v_inception_3_a(vfeat)
        vfeat = self.v_inception_3_b(vfeat)
        vfeat = self.v_maxpool_3(vfeat)
        vfeat = self.v_inception_4_a(vfeat)
        vfeat_shortcut_1 = vfeat
        vfeat = self.v_inception_4_b(vfeat)
        vfeat = self.v_inception_4_c(vfeat)
        vfeat = self.v_inception_4_d(vfeat)
        vfeat_shortcut_2 = vfeat
        vfeat = self.v_inception_4_e(vfeat)
        vfeat = self.v_maxpool_4(vfeat)
        vfeat = self.v_inception_5_a(vfeat)
        vfeat = self.v_inception_5_b(vfeat)
        vfeat = self.v_avgpool_5(vfeat)
        vfeat = vfeat.contiguous()
        vfeat = vfeat.view(-1,1024)
        vfeat_shortcut_1 = self.v_shortcut_1_cov1(vfeat_shortcut_1)
        vfeat_shortcut_1 = self.v_shortcut_1_avgpool(vfeat_shortcut_1)
        vfeat_shortcut_1 = vfeat_shortcut_1.contiguous()
        vfeat_shortcut_1 = vfeat_shortcut_1.view(-1,1024)
        vfeat_shortcut_2 = self.v_shortcut_2_cov1(vfeat_shortcut_2)
        vfeat_shortcut_2 = self.v_shortcut_2_avgpool(vfeat_shortcut_2)
        vfeat_shortcut_2 = vfeat_shortcut_2.contiguous()
        vfeat_shortcut_2 = vfeat_shortcut_2.view(-1,1024)
        vfeat = vfeat + 0.3*vfeat_shortcut_1 + 0.3*vfeat_shortcut_2
        vfeat = self.v_BN_3(vfeat)
        vfeat = F.dropout(p=0.4,input=vfeat, training=True)      #attention::change this to False when eveluate
        vfeat = self.v_FC(vfeat)

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
        afeat = self.a_inception_3_a(afeat)
        afeat = self.a_inception_3_b(afeat)
        afeat = self.a_maxpool_3(afeat)
        afeat = self.a_inception_4_a(afeat)
        afeat_shortcut_1 = afeat
        afeat = self.a_inception_4_b(afeat)
        afeat = self.a_inception_4_c(afeat)
        afeat = self.a_inception_4_d(afeat)
        afeat_shortcut_2 = afeat
        afeat = self.a_inception_4_e(afeat)
        afeat = self.a_maxpool_4(afeat)
        afeat = self.a_inception_5_a(afeat)
        afeat = self.a_inception_5_b(afeat)
        afeat = self.a_avgpool_5(afeat)
        afeat = afeat.contiguous()
        afeat = afeat.view(-1,512)
        afeat_shortcut_1 = self.a_shortcut_1_cov1(afeat_shortcut_1)
        afeat_shortcut_1 = self.a_shortcut_1_avgpool(afeat_shortcut_1)
        afeat_shortcut_1 = afeat_shortcut_1.contiguous()
        afeat_shortcut_1 = afeat_shortcut_1.view(-1,512)
        afeat_shortcut_2 = self.a_shortcut_2_cov1(afeat_shortcut_2)
        afeat_shortcut_2 = self.a_shortcut_2_avgpool(afeat_shortcut_2)
        afeat_shortcut_2 = afeat_shortcut_2.contiguous()
        afeat_shortcut_2 = afeat_shortcut_2.view(-1,512)
        afeat = afeat + 0.3*afeat_shortcut_1 + 0.3*afeat_shortcut_2
        afeat = self.a_BN_3(afeat)
        afeat = F.dropout(p=0.4,input=afeat, training=True)
        afeat = self.a_FC(afeat)
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
        loss = torch.mean((1-label) * torch.pow(dist, 2) +
                (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))

        return loss



























