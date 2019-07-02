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
import sys
from sklearn.preprocessing import normalize

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
        self.level = lev_sphere[5]
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

    def __init__(self, data_dir, train, test, valid, partition="train"):  # data_dir = "/fs4/masi/haol/data/"
        assert (partition in ["train", "test", "validation"])
        lev_sphere = [12, 42, 162, 642, 2562, 10242, 40962, 163842]
        self.level = lev_sphere[5]
        self.demo_num = 2

        if partition == "train":
            trainfile = train['Session'].values.astype(str)
            label = train['dx_group'].values.astype(np.int64)

            # matrix = self.readin(data_dir, '.lh.CT.txt', trainfile)  # accuracy: < 64%, validation 74%, auc:0.5,0.56,0.2
            # matrix_ = self.readin(data_dir, '.rh.CT.txt', trainfile)
            # matrix2 = self.readin(data_dir, '.lh.SD.txt', trainfile)  # accuracy: < 72%, validation 77%, auc:0.83,0.71
            # matrix2_ = self.readin(data_dir, '.rh.SD.txt', trainfile)
            matrix3 = self.readin(data_dir, '.lh.LGI.txt', trainfile)  # accuracy: < 48%, validation 76%, auc: < 0.5
            matrix3_ = self.readin(data_dir, '.rh.LGI.txt', trainfile)
            #matrix4 = self.readin(data_dir, '.lh.SCI.txt', trainfile)  # accuracy: < 50%, validation 60%, auc = 0.5
            #matrix4_ = self.readin(data_dir, '.rh.SCI.txt', trainfile)
            #matrix5 = self.readin(data_dir, '.lh.S.txt', trainfile)  # accuracy: < 61%, validation 67%, auc < 0.38
            #matrix5_ = self.readin(data_dir, '.rh.S.txt', trainfile)
            #matrix6 = self.readin(data_dir, '.lh.H.txt', trainfile) # accuracy: < 50%, validation 68%, auc < 0.67,0.38,0.2
            #matrix6_ = self.readin(data_dir, '.rh.H.txt', trainfile)

            lab = np.expand_dims(label, 1)
            meta = self.demographic(train)
            meta1 = np.reshape(meta, (self.demo_num, -1)).T
            label_mat = np.concatenate((lab, meta1), 1)
            self.y = label_mat
            # self.x = matrix

            x = np.concatenate((matrix3, matrix3_),1)
            self.x = x
            #self.y = label
        elif partition == "test":
            testfile = test['Session'].values.astype(str)
            label = test['dx_group'].values.astype(np.int64)

            # matrix = self.readin(data_dir, '.lh.CT.txt', testfile)
            # matrix_ = self.readin(data_dir, '.rh.CT.txt', testfile)
            # matrix2 = self.readin(data_dir, '.lh.SD.txt', testfile)
            # matrix2_ = self.readin(data_dir, '.rh.SD.txt', testfile)
            matrix3 = self.readin(data_dir, '.lh.LGI.txt', testfile)
            matrix3_ = self.readin(data_dir, '.rh.LGI.txt', testfile)
            #matrix4 = self.readin(data_dir, '.lh.SCI.txt', testfile)
            #matrix4_ = self.readin(data_dir, '.rh.SCI.txt', testfile)
            #matrix5 = self.readin(data_dir, '.lh.S.txt', testfile)
            #matrix5_ = self.readin(data_dir, '.rh.S.txt', testfile)
            # matrix6 = self.readin(data_dir, '.lh.H.txt', testfile)
            # matrix6_ = self.readin(data_dir, '.rh.H.txt', testfile)
            #self.y = label

            lab = np.expand_dims(label, 1)
            meta = self.demographic(test)
            meta1 = np.reshape(meta, (self.demo_num, -1)).T
            label_mat = np.concatenate((lab, meta1), 1)
            self.y = label_mat
            #self.x = matrix

            x = np.concatenate((matrix3, matrix3_), 1)
            #self.y = label
            self.x = x
        else:
            validfile = valid['Session'].values.astype(str)
            label = valid['dx_group'].values.astype(np.int64)

            #matrix = self.readin(data_dir, '.lh.CT.txt', validfile)
            #matrix2 = self.readin(data_dir, '.lh.SD.txt', validfile)
            #matrix2_ = self.readin(data_dir, '.rh.SD.txt', validfile)
            matrix3 = self.readin(data_dir, '.lh.LGI.txt', validfile)
            matrix3_ = self.readin(data_dir, '.rh.LGI.txt', validfile)
            #matrix4 = self.readin(data_dir, '.lh.SCI.txt', validfile)
            #matrix4_ = self.readin(data_dir, '.rh.SCI.txt', validfile)
            #matrix5 = self.readin(data_dir, '.lh.S.txt', validfile)
            #matrix5_ = self.readin(data_dir, '.rh.S.txt', validfile)
            #matrix6 = self.readin(data_dir, '.lh.H.txt', validfile)
            #matrix6_ = self.readin(data_dir, '.rh.H.txt', validfile)

            lab = np.expand_dims(label, 1)
            meta = self.demographic(valid)
            meta1 = np.reshape(meta, (self.demo_num, -1)).T
            label_mat = np.concatenate((lab, meta1), 1)
            self.y = label_mat
            # self.x = matrix

            x = np.concatenate((matrix3, matrix3_), 1)
            # self.y = label
            self.x = x

        print(np.shape(self.x))
        print(np.shape(self.y))

    def readin(self, data_dir, channel_name, session):  # channel_name: '.CT.txt', session: trainfile
        H = Parallel(n_jobs=18)(delayed(self.process)(i, name, data_dir, channel_name) for i, name in enumerate(session))
        H.sort()
        H = [r[1] for r in H]
        H = np.reshape(H, (-1, 163842))
        H = H[:, :self.level].astype(np.float32)
        H = np.expand_dims(H, 1)    # should be expanded if multi-channel
        return H

    def process(self, i, name, data_dir, channel_name):
        if i % 5 == 0:
            print(i)
        f = join(data_dir, name) + channel_name
        return [i, np.loadtxt(fname=f)]

    def demographic(self, table):
        age = table['age_at_scan'].values.astype(np.int64)
        age_norm = normalize(age[:, np.newaxis], axis=0).ravel()
        # re1 = table.replace({'phi_gender': 0}, 2)  # (0, 1) to (2, 1)
        re2 = table.replace({'phi_gender': 0}, -1)  # (0, 1) to (-1, 1)
        sex = re2['phi_gender'].values.astype(np.int64)
        a = table.replace({'Scan': 'Improved 3D'}, 1)
        a = a.replace({'Scan': 'Improved 3D SENSE'}, 2)
        a = a.replace({'Scan': 'T1W'}, 3)
        a = a.replace({'Scan': 'T1W/3D/TFE'}, 4)
        scan = a['Scan'].values.astype(np.int64)
        metadata = np.concatenate((age_norm, sex)).astype(np.float32)
        return metadata
        #return age_norm


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]