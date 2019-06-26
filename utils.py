import torch
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from torch.utils.data import Dataset
import pickle, gzip
import os
import pandas as pd
from os.path import join
from joblib import Parallel, delayed
import multiprocessing

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def sparse2tensor(m):
    """
    Convert sparse matrix (scipy.sparse) to tensor (torch.sparse)
    """
    assert(isinstance(m, sparse.coo.coo_matrix))
    i = torch.LongTensor([m.row, m.col])
    v = torch.FloatTensor(m.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(m.shape))

def spmatmul(den, sp):
    """
    den: Dense tensor of shape batch_size x in_chan x #V
    sp : Sparse tensor of shape newlen x #V
    """
    batch_size, in_chan, nv = list(den.size())
    new_len = sp.size()[0]
    den = den.permute(2, 1, 0).contiguous().view(nv, -1)
    # parvatp - updated below line to support fix data type issue float vs double
    res = torch.spmm(sp, den).view(new_len, in_chan, batch_size).contiguous().permute(2, 1, 0)
    #res = torch.spmm(sp, den.float()).view(new_len, in_chan, batch_size).contiguous().permute(2, 1, 0)
    return res

def xyz2latlong(vertices):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    long = np.arctan2(y, x)
    xy2 = x**2 + y**2
    lat = np.arctan2(z, np.sqrt(xy2))
    return lat, long

def interp_r2tos2(sig_r2, V, method="linear", dtype=np.float32):
    """
    sig_r2: rectangular shape of (lat, long, n_channels)
    V: array of spherical coordinates of shape (n_vertex, 3)
    method: interpolation method. "linear" or "nearest"
    """
    ele, azi = xyz2latlong(V)
    nlat, nlong = sig_r2.shape[0], sig_r2.shape[1]
    dlat, dlong = np.pi/(nlat-1), 2*np.pi/nlong
    lat = np.linspace(-np.pi/2, np.pi/2, nlat)
    long = np.linspace(-np.pi, np.pi, nlong+1)
    sig_r2 = np.concatenate((sig_r2, sig_r2[:, 0:1]), axis=1)
    intp = RegularGridInterpolator((lat, long), sig_r2, method=method)
    s2 = np.array([ele, azi]).T
    sig_s2 = intp(s2).astype(dtype)
    return sig_s2
    
class MNIST_S2_Loader(Dataset):
    """Data loader for spherical MNIST dataset."""

    def __init__(self, data_zip, partition="train"):
        """
        Args:
            data_zip: path to zip file for data
            partition: train or test
        """
        assert(partition in ["train", "test"])
        self.data_dict = pickle.load(gzip.open(data_zip, "rb"))
        if partition == "train":
            self.x = self.data_dict["train_inputs"]/255
            self.y = self.data_dict["train_labels"]
        else:
            self.x = self.data_dict["test_inputs"]/255
            self.y = self.data_dict["test_labels"]
        self.x = (np.expand_dims(self.x, 1) - 0.1307)/0.3081
        print(np.shape(self.x))
        print(self.x.dtype)
        print(self.y.dtype)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Test_Loader(Dataset):
    """Data loader for testing."""

    def __init__(self, data_dir, partition="train"):
        """
        Args:
            data_dir: path to data
            partition: train or test
        """
        assert(partition in ["train", "test", "validation"])
        lev_sphere = [12, 42, 162, 642, 2562, 10242, 40962, 163842]
        self.level = lev_sphere[6]
        if partition == "train":
            label, matrix = self.readin(data_dir, 'CTtrain.txt', '/home/haol/Python Script/CV6/CTtraintable.csv')
            #label2, matrix2 = self.readin(data_dir, 'Htrain.txt', '/home/haol/Python Script/CV2/Htraintable.csv')
            #label3, matrix3 = self.readin(data_dir, 'SDtrain.txt', '/home/haol/Python Script/CV2/SDtraintable.csv')
            #label4, matrix4 = self.readin(data_dir, 'LGItrain.txt', '/home/haol/Python Script/CV1/LGItraintable.csv')
            #label5, matrix5 = self.readin(data_dir, 'Strain.txt', '/home/haol/Python Script/CV2/Straintable.csv')
            #label6, matrix6 = self.readin(data_dir, 'SCItrain.txt', '/home/haol/Python Script/CV3/SCItraintable.csv')
            #x = np.concatenate((matrix, matrix6),1)
            #self.y = label
            #self.x = x

            self.x = matrix
            self.y = label

        elif partition == "test":
            label, matrix = self.readin(data_dir, 'CTtest.txt','/home/haol/Python Script/CV6/CTtesttable.csv')
            #label2, matrix2 = self.readin(data_dir, 'Htest.txt', '/home/haol/Python Script/CV2/Htesttable.csv')
            #label3, matrix3 = self.readin(data_dir, 'SDtest.txt', '/home/haol/Python Script/CV2/SDtesttable.csv')
            #label4, matrix4 = self.readin(data_dir, 'LGItest.txt', '/home/haol/Python Script/CV1/LGItesttable.csv')
            #label5, matrix5 = self.readin(data_dir, 'Stest.txt', '/home/haol/Python Script/CV2/Stesttable.csv')
            #label6, matrix6 = self.readin(data_dir, 'SCItest.txt', '/home/haol/Python Script/CV3/SCItesttable.csv')
            #x = np.concatenate((matrix, matrix6),1)
            #self.y = label
            #self.x = x

            self.x = matrix
            self.y = label

        else:
            label, matrix = self.readin(data_dir, 'CTvalidation.txt', '/home/haol/Python Script/CV6/CTvalidationtable.csv')
            # label2, matrix2 = self.readin(data_dir, 'Hvalidation.txt', '/home/haol/Python Script/CV2/Hvalidationtable.csv')
            # label3, matrix3 = self.readin(data_dir, 'SDvalidation.txt', '/home/haol/Python Script/CV2/SDvalidationtable.csv')
            #label4, matrix4 = self.readin(data_dir, 'LGIvalidation.txt', '/home/haol/Python Script/CV1/LGIvalidationtable.csv')
            # label5, matrix5 = self.readin(data_dir, 'Svalidation.txt', '/home/haol/Python Script/CV2/Svalidationtable.csv')
            #label6, matrix6 = self.readin(data_dir, 'SCIvalidation.txt', '/home/haol/Python Script/CV3/SCIvalidationtable.csv')
            #x = np.concatenate((matrix, matrix6), 1)
            #self.y = label
            #self.x = x

            self.x = matrix
            self.y = label


        #self.x = (np.expand_dims(self.x, 1))
        print(np.shape(self.x))
        print(np.shape(self.y))
        print(self.x.dtype)
        print(self.y.dtype)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def readin(self, data_dir, spec_dir, csv_path):
        label = pd.read_csv(csv_path)['dx_group'].values.astype(np.int64) #[0:20]
        full_dir = data_dir + spec_dir
        with open(full_dir) as f:
            dir = f.readlines()
        d = [w.replace('\n', '') for w in dir]
        H_ = np.array([])
        for i in range(len(d)): #20 len(d)
            if i % 5 == 0:
                print(i)
            H_ = np.concatenate([H_, np.loadtxt(fname=d[i])])
        H_ = np.reshape(H_, (-1, 163842))
        H_ = H_[:, :self.level].astype(np.float32)
        H_ = np.expand_dims(H_, 1)
        return label, H_


class Better_Loader(Dataset):
    """Data loader for dataset."""

    def __init__(self, data_dir, cv, td, asd, partition="train"):  # data_dir = "/fs4/masi/haol/data/"
        """
        Args:
            data_dir: path to the folder containing all datafiles
            partition: train or test
        """
        assert (partition in ["train", "test", "validation"])
        lev_sphere = [12, 42, 162, 642, 2562, 10242, 40962, 163842]
        self.level = lev_sphere[5]

        # table = pd.read_csv('/home/haol/Python Script/updated_table.csv')
        # td = table[table['dx_group'] == 0]
        # asd = table[table['dx_group'] == 1]
        # td = td.sample(frac=1).reset_index(drop=True)
        # asd = asd.sample(frac=1).reset_index(drop=True)
        train, test, valid = self.k_folds(cv, td, asd)

        if partition == "train":
            trainfile = train['Session'].values.astype(str)
            label = train['dx_group'].values.astype(np.int64)
            matrix = self.readin(data_dir, '.lh.CT.txt', trainfile)
            #matrix2 = self.readin(data_dir, '.rh.CT.txt', trainfile)
            self.x = matrix
            self.y = label

            # x = np.concatenate((matrix, matrix2),1)
            # self.y = label
            # self.x = x
        elif partition == "test":
            testfile = test['Session'].values.astype(str)
            label = test['dx_group'].values.astype(np.int64)
            matrix = self.readin(data_dir, '.lh.CT.txt', testfile)
            #matrix2 = self.readin(data_dir, '.rh.CT.txt', testfile)
            self.x = matrix
            self.y = label

            # x = np.concatenate((matrix, matrix2), 1)
            # self.y = label
            # self.x = x
        else:
            validfile = valid['Session'].values.astype(str)
            label = valid['dx_group'].values.astype(np.int64)
            matrix = self.readin(data_dir, '.lh.CT.txt', validfile)
            #matrix2 = self.readin(data_dir, '.rh.CT.txt', validfile)
            self.x = matrix
            self.y = label

            # x = np.concatenate((matrix, matrix2), 1)
            # self.y = label
            # self.x = x

        print(np.shape(self.x))
        print(np.shape(self.y))

    def readin(self, data_dir, channel_name, session):  # channel_name: '.CT.txt', session: trainfile
        H = np.array([])
        for i, name in enumerate(session):
            if i % 5 == 0:
                print(i)
            f = join(data_dir, name) + channel_name
            H = np.concatenate([H, np.loadtxt(fname=f)])
        H = np.reshape(H, (-1, 163842))
        H = H[:, :self.level].astype(np.float32)

        H = np.expand_dims(H, 1)    # should be expanded if multi-channel
        return H

    # def readin(self, data_dir, channel_name, session):  # channel_name: '.CT.txt', session: trainfile
    #     # for i, name in enumerate(session):
    #     #     H = self.process(i, name)
    #     H = [None] * len(session)
    #     #num_cores = multiprocessing.cpu_count()
    #     H = Parallel(n_jobs=18, require='sharedmem')(delayed(self.process)(i, name, data_dir, channel_name, H) for i, name in enumerate(session))
    #
    #     H = np.reshape(H, (-1, 163842))
    #     H = H[:, :self.level].astype(np.float32)
    #     H = np.expand_dims(H, 1)    # should be expanded if multi-channel
    #     return H
    #
    #
    # def process(self, i, name, data_dir, channel_name, H):
    #     if i % 5 == 0:
    #         print(i)
    #     f = join(data_dir, name) + channel_name
    #     H[i] = np.loadtxt(fname=f)
    #     return H

    def k_folds(self, CV, TD, ASD, size_v=1, size_t=1):
        k = 10
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

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]