import numpy as np
from glob import glob
import os
from os.path import join
import random
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
import multiprocessing

# sphere mesh size at different levels
nv_sphere = [12, 42, 162, 642, 2562, 10242, 40962, 163842]
# precomputed mean and std of the dataset
precomp_mean = [0.4974898, 0.47918808, 0.42809588, 1.0961773]
precomp_std = [0.23762763, 0.23354423, 0.23272438, 0.75536704]


class S2D3DSegLoader(Dataset):
    """Data loader for 2D3DS dataset."""

    def __init__(self, data_dir, partition, fold, sp_level, in_ch=3, normalize_mean=precomp_mean,
                 normalize_std=precomp_std):
        """
        Args:
            data_dir: path to data directory
            partition: train or test
            fold: 1, 2 or 3 (for 3-fold cross-validation)
            sp_level: sphere mesh level. integer between 0 and 7.

        """
        assert (partition in ["train", "test"])
        assert (fold in [1, 2, 3])
        self.in_ch = in_ch
        self.nv = nv_sphere[sp_level]
        self.partition = partition

        flist1 = []
        file_format1 = os.path.join(data_dir, "*.lh.*.txt")
        flist1 += sorted(glob(file_format1))

        # dict constuction
        data = dict()
        for i in flist1:
            key = i.split('.')[0].split('/')[-1]
            if not key in data:
                data[key] = {'subject': key}
            data[key].setdefault(i.split('.')[2], []).append(i)

        # subject list
        subj = [entry for entry in data]
        subj = sorted(subj)

        # cross validation
        train = subj[0:48]
        val = subj[48:54]
        test = subj[54:60]

        self.flist = []
        # final list
        if partition == "train":
            flist_train = []
            for i in train:
                for j in range(0, 16):
                    flist_train.append(data[i]['reg' + str(j)])
            self.flist = flist_train
        if partition == "validation":
            flist_val = []
            for i in val:
                for j in range(0, 1):
                    flist_val.append(data[i]['reg' + str(j)])
            self.flist = flist_val
        if partition == "test":
            flist_test = []
            for i in test:
                flist_test.append(data[i]['reg' + str(15)])
            self.flist = flist_test

        # self.mean = np.expand_dims(precomp_mean, -1).astype(np.float32)
        # self.std = np.expand_dims(precomp_std, -1).astype(np.float32)

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        # load files
        subj = self.flist[idx]
        feat_filelist = [subj[0], subj[1], subj[3]]
        data = self.readin(feat_filelist).astype(np.float32)
        labels = np.loadtxt(subj[2]).T[:self.nv].astype(np.int)
        fname = subj[0].split('.')[0].split('/')[-1]
        reg = subj[0].split('.')[2]
        return data, labels, fname, reg

    def readin(self, feat_filelist):
        H = Parallel(n_jobs=10)(delayed(self.process)(i, name) for i, name in enumerate(feat_filelist))
        H.sort()
        H = [r[1] for r in H]
        H = np.reshape(H, (-1, 163842))
        H = H[:self.in_ch, :self.nv].astype(np.float32)
        return H

    def process(self, i, file):
        return [i, np.loadtxt(fname=file)]
