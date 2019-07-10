import math
import argparse
import sys

sys.path.append("../../meshcnn")
import numpy as np

np.set_printoptions(precision=4)
import os
import shutil
import logging
from collections import OrderedDict
from tabulate import tabulate

from cpuLoader import S2D3DSegLoader
from model import SphericalUNet

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import scipy
from cpuwrapper import WrappedModel

classes = [101, 103, 105, 107, 109, 113, 115, 117, 119, 121, 123, 125, 129, 133, 135, 137, 139, 141, 143, 145, 147, 149,
           151, 153, 155, 157, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 191, 193, 195, 197,
           199, 201, 203, 205, 207]
classes1 = range(0, 208)
class_names = ["Backgroud", "CS", "SFS", "STS", "IFS", "OTS", "CalcS", "OPS", "CingS", "OLF", "Label10", "Label11",
               "Label12", "Label13", "Label14"]
# class_names = ["unknown", "beam", "board", "bookcase", "ceiling", "chair", "clutter", "column",
#               "door", "floor", "sofa", "table", "wall", "window", "invalid"]
# drop = [0, 14]
drop = [range(0, 100)]
# drop=list(range(0,99))+list(np.array(classes)-1)

keep = np.setdiff1d(classes, drop)
label_ratio = [0.04233976974675504, 0.014504436907968913, 0.017173225930738712,
               0.048004778186652164, 0.17384037404789865, 0.028626771620973622,
               0.087541966989014, 0.019508096683310605, 0.08321331842901526,
               0.17002664771895903, 0.002515611224467519, 0.020731298851232174,
               0.2625963729249342, 0.016994731594287146, 0.012382599143792165]
# label_weight = 1/np.array(label_ratio)/np.sum((1/np.array(label_ratio))[keep])
label_weight = 1 / np.log(1.02 + np.array(label_ratio))
# label_weight[drop] = 0
label_weight = label_weight.astype(np.float32)


def save_checkpoint(state, is_best, epoch, output_folder, filename, logger):
    if epoch > 1:
        os.remove(output_folder + filename + '_%03d' % (epoch - 1) + '.pth.tar')
    torch.save(state, output_folder + filename + '_%03d' % epoch + '.pth.tar')
    if is_best:
        logger.info("Saving new best model")
        shutil.copyfile(output_folder + filename + '_%03d' % epoch + '.pth.tar',
                        output_folder + filename + '_best.pth.tar')


def iou_score(pred_cls, true_cls, nclass=207, drop=drop):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    intersect_ = []
    union_ = []
    for i in range(nclass):
        if i not in drop:
            intersect = ((pred_cls == i) + (true_cls == i)).eq(2).sum().item()
            union = ((pred_cls == i) + (true_cls == i)).ge(1).sum().item()
            intersect_.append(intersect)
            union_.append(union)
    return np.array(intersect_), np.array(union_)


def accuracy(pred_cls, true_cls, nclass=207, drop=drop):
    positive = torch.histc(true_cls.cpu().float(), bins=nclass, min=0, max=nclass, out=None)
    per_cls_counts = []
    tpos = []
    for i in range(nclass):
        if i not in drop:
            true_positive = ((pred_cls == i) + (true_cls == i)).eq(2).sum().item()
            tpos.append(true_positive)
            per_cls_counts.append(positive[i])
    return np.array(tpos), np.array(per_cls_counts)


def dice_loss(pred, target):
    """

    """
    smooth = 1
    iflat = torch.from_numpy(pred).contiguous().view(-1)
    tflat = torch.from_numpy(target).contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / A_sum + B_sum + smooth)


def iou_dicescore(pred_cls, true_cls, nclass=207, drop=drop):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    intersect_ = []
    union_ = []
    for i in range(nclass):
        if i not in drop:
            intersect = ((pred_cls == i) + (true_cls == i)).eq(2).sum().item()
            union = ((pred_cls == i) + (true_cls == i)).ge(1).sum().item()
            intersect_.append(intersect)
            union_.append(union)
    return torch.div(torch.FloatTensor(intersect_), torch.FloatTensor(union_))


def train(args, model, train_loader, optimizer, epoch, device, logger, keep_id=None):
    w = torch.tensor(label_weight).to(device)
    model.train()
    tot_loss = 0
    count = 0
    for batch_idx, (data, target, fname) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        if keep_id is not None:
            output = output[:, :, keep_id]
            target = target[:, keep_id]

        # loss = F.cross_entropy(output, target, weight=w)
        loss = F.cross_entropy(output, target)
        # loss=dice_loss_3d(torch.from_numpy(pred_new),torch.from_numpy(target_new))
        # loss=dice_loss_3d(output,target)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        count += data.size()[0]
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        pred = output.max(dim=1, keepdim=False)[1]
        scipy.io.savemat(os.path.join(args.log_dir, "trainDataParcel.mat"),
                         {'fname': fname,'output': pred.cuda().detach().cpu().numpy(),
                          'prob': output.cuda().detach().cpu().numpy(),
                          'target': target.cuda().detach().cpu().numpy(), 'data': data.cuda().detach().cpu().numpy()})
    tot_loss /= count
    return tot_loss


def test(args, model, test_loader, epoch, device, logger, keep_id=None):
    w = torch.tensor(label_weight).to(device)
    model.eval()
    test_loss = 0
    ints_ = np.zeros(len(classes1) - len(drop))
    unis_ = np.zeros(len(classes1) - len(drop))
    per_cls_counts = np.zeros(len(classes1) - len(drop))
    accs = np.zeros(len(classes1) - len(drop))
    count = 0
    with torch.no_grad():
        for data, target, fname, reg in test_loader:
            #data, target = data.cuda(), target.cuda()
            output = model(data)
            n_data = data.size()[0]

            if keep_id is not None:
                output = output[:, :, keep_id]
                target = target[:, keep_id]

            #  test_loss += F.cross_entropy(output, target, weight=w).item() # sum up batch loss
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            pred = output.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
            int_, uni_ = iou_score(pred, target)
            tpos, pcc = accuracy(pred, target)
            ints_ += int_
            unis_ += uni_
            accs += tpos
            per_cls_counts += pcc
            count += n_data
    ious = ints_ / unis_
    accs /= per_cls_counts
    #print(ints_, 'ints')
    #print(unis_, 'unis')
    #print(per_cls_counts, 'per_cls_counts')

    test_loss /= count
    # print(per_cls_counts)

    logger.info('[Epoch {} {} stats]: MIoU: {:.4f}; Mean Accuracy: {:.4f}; Avg loss: {:.4f}'.format(
        epoch, test_loader.dataset.partition, np.mean(ious), np.mean(accs), test_loss))

    # scipy.io.savemat(os.path.join(args.log_dir, "test.mat"), # "testDataParcel.mat"
    #                  {'fname': fname,'output': pred.cuda().detach().cpu().numpy(),
    #                   'reg': reg,
    #                   'prob': output.cuda().detach().cpu().numpy(),
    #                   'target': target.cuda().detach().cpu().numpy(),
    #                   'data': data.cuda().detach().cpu().numpy().astype('float')})

    scipy.io.savemat(os.path.join(args.log_dir, "Reg15_" +"Epoch"+ str(epoch) +".mat"),
                     {'fname': fname,
                      'reg': reg,
                      'prob': output.numpy()})

    return np.mean(np.mean(ious))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--mesh_folder', type=str, default="../../mesh_files",
                        help='path to mesh folder (default: ../../mesh_files)')
    parser.add_argument('--data_folder', type=str, default="processed_data",
                        help='path to data folder (default: processed_data)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--max_level', type=int, default=7, help='max mesh level')
    parser.add_argument('--min_level', type=int, default=0, help='min mesh level')
    parser.add_argument('--feat', type=int, default=4, help='filter dimensions')
    parser.add_argument('--log_dir', type=str, default="log",
                        help='log directory for run')
    parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
    parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--resume', type=str, default=None, help="path to checkpoint if resume is needed")
    parser.add_argument('--fold', type=int, choices=[1, 2, 3], required=True,
                        help="choice among 3 fold for cross-validation")
    parser.add_argument('--blackout_id', type=str, default="", help="path to file storing blackout_id")
    parser.add_argument('--in_ch', type=str, default="rgbd", choices=["rgb", "rgbd", "rgbdab"], help="input channels")
    parser.add_argument('--train_stats_freq', default=0, type=int,
                        help="frequency for printing training set stats. 0 for never.")

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # logger and snapshot current code
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    shutil.copy2(__file__, os.path.join(args.log_dir, "script.py"))
    shutil.copy2("model.py", os.path.join(args.log_dir, "model.py"))
    shutil.copy2("run.sh", os.path.join(args.log_dir, "run.sh"))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(args.log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("%s", repr(args))

    torch.manual_seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

#    trainset = S2D3DSegLoader(args.data_folder, "train", fold=args.fold, sp_level=args.max_level, in_ch=len(args.in_ch))
#    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    valset = S2D3DSegLoader(args.data_folder, "test", fold=args.fold, sp_level=args.max_level, in_ch=len(args.in_ch))
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = SphericalUNet(mesh_folder=args.mesh_folder, in_ch=len(args.in_ch), out_ch=20,
                          max_level=args.max_level, min_level=args.min_level, fdim=args.feat)

    #FIXME in GPU mode. CPU does not require DaraParallel(), use WrappedModel instread
    #model = nn.DataParallel(model)
    model = WrappedModel(model)

    model.to(device)

    if args.blackout_id:
        blackout_id = np.load(args.blackout_id)
        keep_id = np.argwhere(np.isin(np.arange(model.module.nv_max), blackout_id, invert=True))
    else:
        keep_id = None

    start_ep = 0
    best_miou = 0
    checkpoint_path = os.path.join('/nfs/masi/lyui/Sulcus/NORA_PFCSulci/parc/reflected/', 'checkpoint_latest.pth.tar')
    if args.resume:
        for i in range(1, 101):
            #resume_dict = torch.load(args.resume)
            #start_ep = resume_dict['epoch']
            #best_miou = resume_dict['best_miou']
            resume_dict = torch.load(checkpoint_path + '_SUNet' + '_%03d' % i + '.pth.tar')

            def load_my_state_dict(self, state_dict, exclude='none'):
                from torch.nn.parameter import Parameter

                own_state = self.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    if exclude in name:
                        continue
                    if isinstance(param, Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    own_state[name].copy_(param)

            load_my_state_dict(model, resume_dict['state_dict'])
            test(args, model, val_loader, i, device, logger, keep_id)


    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)



if __name__ == "__main__":
    main()
