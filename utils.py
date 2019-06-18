import torch
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from torch.utils.data import Dataset
import pickle, gzip
import os
import pandas as pd

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
            label, matrix = self.readin(data_dir, 'CTtrain.txt', '/home/haol/Python Script/CTtraintable.csv')
            #label2, matrix2 = self.readin(data_dir, 'Htrain.txt', '/home/haol/Python Script/Htraintable.csv')
            #label3, matrix3 = self.readin(data_dir, 'SDtrain.txt', '/home/haol/Python Script/SDtraintable.csv')
            #label4, matrix4 = self.readin(data_dir, 'LGItrain.txt', '/home/haol/Python Script/LGItraintable.csv')
            #label5, matrix5 = self.readin(data_dir, 'Strain.txt', '/home/haol/Python Script/Straintable.csv')
            #label6, matrix6 = self.readin(data_dir, 'SCItrain.txt', '/home/haol/Python Script/SCItraintable.csv')
            # x = np.concatenate((matrix, matrix6),1)
            # self.y = label
            # self.x = x

            self.x = matrix
            self.y = label

        elif partition == "test":
            label, matrix = self.readin(data_dir, 'CTtest.txt','/home/haol/Python Script/CTtesttable.csv')
            #label2, matrix2 = self.readin(data_dir, 'Htest.txt', '/home/haol/Python Script/Htesttable.csv')
            #label3, matrix3 = self.readin(data_dir, 'SDtest.txt', '/home/haol/Python Script/SDtesttable.csv')
            #label4, matrix4 = self.readin(data_dir, 'LGItest.txt', '/home/haol/Python Script/LGItesttable.csv')
            #label5, matrix5 = self.readin(data_dir, 'Stest.txt', '/home/haol/Python Script/Stesttable.csv')
            # label6, matrix6 = self.readin(data_dir, 'SCItest.txt', '/home/haol/Python Script/SCItesttable.csv')
            # x = np.concatenate((matrix, matrix6),1)
            # self.y = label
            # self.x = x

            self.x = matrix
            self.y = label

        else:
            label, matrix = self.readin(data_dir, 'CTvalidation.txt', '/home/haol/Python Script/CTvalidationtable.csv')
            # label2, matrix2 = self.readin(data_dir, 'Hvalidation.txt', '/home/haol/Python Script/Hvalidationtable.csv')
            # label3, matrix3 = self.readin(data_dir, 'SDvalidation.txt', '/home/haol/Python Script/SDvalidationtable.csv')
            # label4, matrix4 = self.readin(data_dir, 'LGIvalidation.txt', '/home/haol/Python Script/LGIvalidationtable.csv')
            # label5, matrix5 = self.readin(data_dir, 'Svalidation.txt', '/home/haol/Python Script/Svalidationtable.csv')
            # label6, matrix6 = self.readin(data_dir, 'SCIvalidation.txt', '/home/haol/Python Script/SCIvalidationtable.csv')
            # x = np.concatenate((matrix, matrix6), 1)
            # self.y = label
            # self.x = x

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
        #label = np.expand_dims(label, 1)
        full_dir = data_dir + spec_dir
        with open(full_dir) as f:
            dir = f.readlines()
        d = [w.replace('\n', '') for w in dir]
        H_ = np.array([])
        for i in range(len(d)): #20
            if i % 5 == 0:
                print(i)
            H_ = np.concatenate([H_, np.loadtxt(fname=d[i])])
        H_ = np.reshape(H_, (-1, 163842))
        H_ = H_[:, :self.level].astype(np.float32)
        H_ = np.expand_dims(H_, 1)
        return label, H_