import math
import argparse
import sys, os; sys.path.append("../../meshcnn")
import numpy as np
import pickle, gzip
import logging
import shutil
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import normalize

from utils import sparse2tensor, spmatmul, MNIST_S2_Loader, Test_Loader, Better_Loader
from ops import MeshConv
from model import Model, DeformModel2, mod

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import nn
from collections import OrderedDict


def train(args, model, device, train_loader, optimizer, epoch, logger):
    model.train()
    train_loss = 0
    correct = 0
    batch = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # target1 = target[:, 0]
        # metadata = target[:, 1:].numpy().astype(np.float32)
        # data, target = data.to(device), target1.flatten().to(device) # data is input data, target is labels
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        metadata = []
        output = model(data, metadata)

        pred = output.clone()
        for i in range(len(output)):
            if output[i] >= 0.5:
                pred[i] = 1
            else:
                pred[i] = 0
        target = target.float()

        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = nn.BCELoss(reduction='mean')
        loss = loss(output, target)
        train_loss += loss
        loss.backward()
        optimizer.step()
        batch += 1

    train_loss /= batch
    logger.info('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) \r'.format(
                train_loss, correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))
    state_dict_no_sparse = [it for it in model.state_dict().items() if
                            it[1].type() != "torch.cuda.sparse.FloatTensor"]
    state_dict_no_sparse = OrderedDict(state_dict_no_sparse)
    state = {'state_dict': state_dict_no_sparse}
    torch.save(state, '/home/haol/Python_Script/model/' + str(epoch) + '.pth')
    #return output, target


def test(args, model, device, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    batch = 0
    with torch.no_grad():
        for data, target in test_loader:
            # target1 = target[:, 0]
            # metadata = target[:, 1:].numpy().astype(np.float32)
            # data, target = data.to(device), target1.flatten().to(device)
            # data, target[0, :] = data.to(device), target.to(device)
            data, target = data.to(device), target.to(device)
            metadata = []
            output = model(data, metadata)

        pred = output.clone()
        for i in range(len(output)):
            if output[i] >= 0.5:
                pred[i] = 1
            else:
                pred[i] = 0
        target = target.float()

        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = nn.BCELoss(reduction='mean')
        loss = loss(output, target)
        test_loss += loss
        batch += 1

    test_loss /= batch
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) \r'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
    #return output, target

def validation(args, model, device, valid_loader, logger):
    model.eval()
    valid_loss = 0
    correct = 0
    batch = 0
    with torch.no_grad():
        for data, target in valid_loader:
            # target1 = target[:, 0]
            # metadata = target[:, 1:].numpy().astype(np.float32)
            # data, target = data.to(device), target1.flatten().to(device)
            # data, target[0, :] = data.to(device), target.to(device)
            # metadata = np.array([[0,0,0]] * len(data), dtype=np.float32)
            data, target = data.to(device), target.to(device)
            metadata = []
            output = model(data, metadata)

        pred = output.clone()
        for i in range(len(output)):
            if output[i] >= 0.5:
                pred[i] = 1
            else:
                pred[i] = 0
        target = target.float()

        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = nn.BCELoss(reduction='mean')
        loss = loss(output, target)
        valid_loss += loss
        batch += 1

    valid_loss /= batch
    logger.info('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) \r'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    #return valid_loss
   #return 100. * correct / len(valid_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--mesh_folder', type=str, required=True,
                        help='path to mesh folder (default: mesh_files)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--datafile', type=str, default="mnist_ico4.gzip",
                        help='data file containing preprocessed spherical mnist data')
    parser.add_argument('--log_dir', type=str, default="log",
                        help='log directory for run')
    parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
    parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--feat', type=int, default=16, help="number of base features")

    args = parser.parse_args()

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

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

#################################################################################################################
    table = pd.read_csv('/home/haol/Python_Script/updated_table.csv')
    td = table[table['dx_group'] == 0]
    asd = table[table['dx_group'] == 1]
    #td = td.reset_index(drop=True)
    #asd = asd.reset_index(drop=True)
    td = td.sample(frac=1).reset_index(drop=True)
    asd = asd.sample(frac=1).reset_index(drop=True)


    for j in range(3):  # j is cv in range 5
        acc = []
        lst = []

        traintb, testtb, validtb = k_folds(j, td, asd)

        trainset = Better_Loader(args.datafile, traintb, testtb, validtb, "train")
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, **kwargs)
        testset = Better_Loader(args.datafile, traintb, testtb, validtb, "test")
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
        validset = Better_Loader(args.datafile, traintb, testtb, validtb, "validation")
        valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, **kwargs)

        model = DeformModel2(mesh_folder=args.mesh_folder, feat=args.feat)
        model = nn.DataParallel(model).cuda()
        model.to(device)

        logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

        if args.optim == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if args.decay:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

        for epoch in range(1, args.epochs + 1):  # args.epochs + 1
            if args.decay:
                scheduler.step()
            logger.info("[Epoch {}]".format(epoch))
            train(args, model, device, train_loader, optimizer, epoch, logger)
            #train(args, model, device, train_loader, optimizer, epoch, logger)
            #acc.append(validation(args, model, device, valid_loader, logger))
            validation(args, model, device, valid_loader, logger)
            #lst.append(test(args, model, device, test_loader, logger))
            test(args, model, device, test_loader, logger)

        # idx = acc.index(max(acc))
        # print('Epoch:'+ str(idx + 1) + ' '+ 'Validation Accuracy:'+ str(acc[idx]))
        # #output, target = lst[idx]

##########################################################################
#         resume_dict = torch.load('/home/haol/Python_Script/model/' + str(idx+1)+'.pth')  # best accuracy in validation
#
#         def load_my_state_dict(self, state_dict, exclude='none'):
#             from torch.nn.parameter import Parameter
#             own_state = self.state_dict()
#             for name, param in state_dict.items():
#                 if name not in own_state:
#                     continue
#                 if exclude in name:
#                     continue
#                 if isinstance(param, Parameter):
#                     # backwards compatibility for serialized parameters
#                     param = param.data
#                 own_state[name].copy_(param)
#
#         load_my_state_dict(model, resume_dict['state_dict'])
#
#         output, target = train(args, model, device, train_loader, optimizer, epoch, logger)
#         out, tar = test(args, model, device, test_loader, None)
#
#         age1, sex1, scan1 = demographic(traintb)
#         age2, sex2, scan2 = demographic(testtb)
#
#         # combine age, sex, scan and out for test
#         output = output.detach()
#         output = output.flatten()
#         X_train = np.concatenate((output.cpu().numpy(), age1, sex1, scan1))
#         y_train = target.cpu().numpy()
#         X_train = np.reshape(X_train, (len(y_train),-1))
#         print(X_train)
#         out = out.flatten()
#         X_test = np.concatenate((out.cpu().numpy(), age2, sex2, scan2)) # 4 is the number of demo info to combine
#         y_test = tar.cpu().numpy()
#         X_test = np.reshape(X_test, (len(y_test),-1))
#         print(X_test)
#
#         clf = SVC(gamma='auto', probability=True)
#         clf.fit(X_train, y_train)
#         a = clf.score(X_test, y_test)
#         print(str(a)+' ' + 'accuracy')
#         prob = clf.predict_proba(X_test)
#         fpr, tpr, thresholds = roc_curve(tar.cpu().numpy(), prob[:, 1], drop_intermediate=False)
# ##########################################################################
#
#         #fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), output.cpu().numpy(), drop_intermediate=False)  # print(fpr,tpr)
#         au = auc(fpr, tpr)
#         plt.plot(fpr, tpr, lw=1, alpha=0.3,
#                  label='ROC fold %d (AUC = %0.2f)' % (j, au))
#         print(str(au)+' ' + 'auc')
#
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC curves')
#     plt.legend(loc="lower right")
#     plt.show()



def k_folds(CV, TD, ASD, size_v=1, size_t=1):
    k = 5
    S_ASD = int(len(ASD) / k)
    S_TD = int(len(TD) / k)
    P_ASD = []
    P_TD = []
    for i in range(k + 1):
        P_ASD.append(i * S_ASD)
        P_TD.append(i * S_TD)
    P_ASD[-1] = len(ASD)
    P_TD[-1] = len(TD)
    frames = []
    frames2 = []
    frames3 = []
    for i in range(k - size_v - size_t):
        idx1 = (CV + i) % k
        idx2 = idx1 + 1
        a = ASD.loc[range(P_ASD[idx1], P_ASD[idx2])]
        b = TD.loc[range(P_TD[idx1], P_TD[idx2])]
        frames.append(a)
        frames.append(b)
    train = pd.concat(frames, ignore_index=False)
    for i in range(size_t):
        idx3 = (idx2 + i) % k
        idx4 = idx3 + 1
        c = ASD.loc[range(P_ASD[idx3], P_ASD[idx4])]
        d = TD.loc[range(P_TD[idx3], P_TD[idx4])]
        frames2.append(c)
        frames2.append(d)
    test = pd.concat(frames2, ignore_index=False)
    for i in range(size_v):
        idx5 = (idx4 + i) % k
        idx6 = idx5 + 1
        e = ASD.loc[range(P_ASD[idx5], P_ASD[idx6])]
        f = TD.loc[range(P_TD[idx5], P_TD[idx6])]
        frames3.append(e)
        frames3.append(f)
    valid = pd.concat(frames3, ignore_index=False)
    return train, test, valid


def test_main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--mesh_folder', type=str, required=True,
                        help='path to mesh folder (default: mesh_files)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--datafile', type=str, default="mnist_ico4.gzip",
                        help='data file containing preprocessed spherical mnist data')
    parser.add_argument('--log_dir', type=str, default="log",
                        help='log directory for run')
    parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
    parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--feat', type=int, default=16, help="number of base features")
    args = parser.parse_args()
    use_cuda = True
    table = pd.read_csv('/home/haol/Python_Script/updated_table.csv')
    td = table[table['dx_group'] == 0][:10]
    asd = table[table['dx_group'] == 1][:10]
    td = td.sample(frac=1).reset_index(drop=True)
    asd = asd.sample(frac=1).reset_index(drop=True)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    testset = Better_Loader(args.datafile, 1, td, asd, "test")
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, **kwargs)

    model = DeformModel2(mesh_folder=args.mesh_folder, feat=args.feat)
    model = nn.DataParallel(model).cuda()
    model = model.to(device)

#  Combine the demographics info
##############################################################################################
    resume_dict = torch.load('/home/haol/Python_Script/model/8.pth') # best accuracy in validation
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

    out, tar = test(args, model, device, test_loader, None)
    #print (out)

    age = table['age_at_scan'].values.astype(np.int64)[:4]
    re1 = table.replace({'phi_gender': 0}, 2)  # (0, 1) to (2, 1)
    # re2 = table.replace({'phi_gender': 0}, -1)  # (0, 1) to (-1, 1)
    sex = re1['phi_gender'].values.astype(np.int64)[:4]
    a = table.replace({'Scan': 'Improved 3D'}, 1)
    a = a.replace({'Scan': 'Improved 3D SENSE'}, 2)
    a = a.replace({'Scan': 'T1W'}, 3)
    a = a.replace({'Scan': 'T1W/3D/TFE'}, 4)
    scan = a['Scan'].values.astype(np.int64)[:4]

    # combine age, sex, scan and out for test
    out = out.flatten()
    X = np.concatenate((out.cpu().numpy(), age, sex, scan))
    X = np.reshape(X, (4, -1)) # 4 is the number of demo info to combine
    y = tar.cpu().numpy()
    print(X)
    clf = SVC(gamma='auto', probability=True)
    clf.fit(X, y)
    a= clf.score(X, y)
    print(a)
    prob = clf.predict_proba(X)

    fpr, tpr, thresholds = roc_curve(tar.cpu().numpy(), prob[:, 1], drop_intermediate=False)
    #au = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.show()




if __name__ == "__main__":
   main()
   #test_main()
