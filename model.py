import torch
import torch.nn as nn
import torch.nn.functional as F
import sys; sys.path.append("../../meshcnn")
from ops import MeshConv, DownSamp, ResBlock
import os

class DownSamp(nn.Module):
    def __init__(self, nv_prev):
        super().__init__()
        self.nv_prev = nv_prev

    def forward(self, x):
        return x[..., :self.nv_prev]

class ResBlock(nn.Module):
    def __init__(self, in_chan, neck_chan, out_chan, level, coarsen, mesh_folder):
        super().__init__()
        l = level-1 if coarsen else level
        self.coarsen = coarsen
        mesh_file = os.path.join(mesh_folder, "icosphere_{}.pkl".format(l))
        self.conv1 = nn.Conv1d(in_chan, neck_chan, kernel_size=1, stride=1)
        self.conv2 = MeshConv(neck_chan, neck_chan, mesh_file=mesh_file, stride=1)
        self.conv3 = nn.Conv1d(neck_chan, out_chan, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(neck_chan)
        self.bn2 = nn.BatchNorm1d(neck_chan)
        self.bn3 = nn.BatchNorm1d(out_chan)
        self.nv_prev = self.conv2.nv_prev
        self.down = DownSamp(self.nv_prev)
        self.diff_chan = (in_chan != out_chan)

        if coarsen:
            self.seq1 = nn.Sequential(self.down, self.conv1, self.bn1, self.relu, 
                                      self.conv2, self.bn2, self.relu, 
                                      self.conv3, self.bn3)
        else:
            self.seq1 = nn.Sequential(self.conv1, self.bn1, self.relu, 
                                      self.conv2, self.bn2, self.relu, 
                                      self.conv3, self.bn3)

        if self.diff_chan:
            self.conv_ = nn.Conv1d(in_chan, out_chan, kernel_size=1, stride=1)
            self.bn_ = nn.BatchNorm1d(out_chan)
            if coarsen:
                self.seq2 = nn.Sequential(self.conv_, self.down, self.bn_)
            else:
                self.seq2 = nn.Sequential(self.conv_, self.bn_)

    def forward(self, x):
        if self.diff_chan:
            x2 = self.seq2(x)
        else:
            x2 = x
        x1 = self.seq1(x)
        out = x1 + x2
        out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, mesh_folder, feat=16, nclasses=10):
        super().__init__()
        mf = os.path.join(mesh_folder, "icosphere_4.pkl")
        self.in_conv = MeshConv(1, feat, mesh_file=mf, stride=1)
        self.in_bn = nn.BatchNorm1d(feat)
        self.relu = nn.ReLU(inplace=True)
        self.in_block = nn.Sequential(self.in_conv, self.in_bn, self.relu)
        self.block1 = ResBlock(in_chan=feat, neck_chan=feat, out_chan=4*feat, level=4, coarsen=True, mesh_folder=mesh_folder)
        self.block2 = ResBlock(in_chan=4*feat, neck_chan=4*feat, out_chan=16*feat, level=3, coarsen=True, mesh_folder=mesh_folder)
        self.avg = nn.AvgPool1d(kernel_size=self.block2.nv_prev) # output shape batch x channels x 1
        self.out_layer = nn.Linear(16*feat, nclasses)

    def forward(self, x):
        x = self.in_block(x)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.squeeze(self.avg(x))
        x = F.dropout(x, training=self.training)
        x = self.out_layer(x)

        return F.log_softmax(x, dim=1)


class DeformModel2(nn.Module):
    # nclasses=363 - ALL
    # step5
    def __init__(self, mesh_folder, feat=16, nclasses=1, n_metadata_features=2):
        super().__init__()
        mf = os.path.join(mesh_folder, "icosphere_5.pkl")
        self.in_conv = MeshConv(2, feat, mesh_file=mf, stride=2)
        self.in_bn = nn.BatchNorm1d(feat)
        self.relu = nn.ReLU(inplace=True)
        self.in_block = nn.Sequential(self.in_conv, self.in_bn, self.relu)

        self.block1 = ResBlock(in_chan=feat, neck_chan=feat, out_chan=4 * feat, level=3, coarsen=True,
                               mesh_folder=mesh_folder)
        self.block2 = ResBlock(in_chan=4 * feat, neck_chan=4 * feat, out_chan=16 * feat, level=2, coarsen=True,
                               mesh_folder=mesh_folder)
        self.avg = nn.AvgPool1d(kernel_size=self.block2.nv_prev)  # output shape batch x channels x 1
        self.resnet_outlayer = nn.Linear(16 * feat, nclasses)  # replaced: nclasses

        # self.block1 = ResBlock(in_chan=feat, neck_chan=feat, out_chan=4 * feat, level=4, coarsen=True,
        #                        mesh_folder=mesh_folder)
        # self.block2 = ResBlock(in_chan=4 * feat, neck_chan=4 * feat, out_chan=16 * feat, level=3, coarsen=True,
        #                        mesh_folder=mesh_folder)
        # self.block3 = ResBlock(in_chan=16 * feat, neck_chan=16 * feat, out_chan=64 * feat, level=2, coarsen=True,
        #                        mesh_folder=mesh_folder)
        # self.avg = nn.AvgPool1d(kernel_size=self.block3.nv_prev)  # output shape batch x channels x 1
        # self.out_layer = nn.Linear(64 * feat, nclasses)

        # Metadata feature layers
        # self.metadata_dense_1 = nn.Linear(n_metadata_features, nclasses)

        # after concatenation
        self.final_outlayer = nn.Linear(n_metadata_features + 1, nclasses)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = self.in_block(x)
        x = self.block1(x)
        x = self.block2(x)

        # x = self.block3(x)
        # x = torch.squeeze(self.avg(x), dim=-1)

        x = torch.squeeze(self.avg(x))

        #x = F.dropout(x, training=self.training)

        x = self.resnet_outlayer(x)
        y = torch.from_numpy(y).cuda()
        x = torch.cat((x, y), 1)
        x = self.final_outlayer(x)
        x = self.sigmoid(x)

        #return F.softmax(x, dim=1)
        return x



class mod(nn.Module):
    def __init__(self, mesh_folder, feat =16,   nclasses=1):
        super().__init__()
        mf = os.path.join(mesh_folder, "icosphere_4.pkl")

        #self.conv1d = nn.Conv1d(in_chan,16, 3 ,stride = 1)
        self.fc = nn.Linear(1284, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(2)
        #mf = os.path.join(mesh_folder, "icosphere_4.pkl")
        self.in_conv = MeshConv(2, 2, mesh_file=mf, stride=2)
        self.in_bn = nn.BatchNorm1d(feat)
        self.relu = nn.ReLU(inplace=True)
        self.in_block = nn.Sequential(self.in_conv,  self.relu)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        batch_size = x.shape[0]
        x = self.in_block(x)
        # x = self.conv1d(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        print (x.shape)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

