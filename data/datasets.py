import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np


class JHUBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):

        #index_pair = np.random.permutation(len(self.paths)) [0:2]
        path = self.paths[index]
        len_files = len( self.paths )
        if index == len_files - 1:
            index1 = 0
        else:
            index1 = index + 1
        #print( "index_pair", index_pair )
        x = pkload(self.paths[index])
        y = pkload(self.paths[index1])

        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        #print("x.shape", x.shape)
        #print( "y.shape", y.shape )
        return x, y

    def __len__(self):
        return len(self.paths)

'''
class JHUBrainInferDataset(Dataset): # 不连续来使用测试集，是12,34,56不是12,23,34
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        #print(self.paths)
        path = self.paths[index]
        len_files = len( self.paths )
        if index >= len_files/2:
            index = index - len_files/2
            index = int(index)
        else:
            index1 = index + 1
        #x, x_seg = pkload(path)
        index = index * 2 
        index1 = index + 1
        #print(index, index1)
        #print(index, index1, "self.paths[index index1]：", self.paths[index], self.paths[index1])
        first = pkload(self.paths[index])
        x = first[0]
        x_seg = first[1]
        second = pkload(self.paths[index1])
        y = second[0]
        y_seg = second[1]
        #print( "self.paths[index]：", self.paths[index], "self.paths[index index]：", self.paths[index1] )

        x, y = x[None, ...], y[None, ...]

        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]

        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x,  x_seg = torch.from_numpy(x),  torch.from_numpy(x_seg)
        y,  y_seg = torch.from_numpy(y),  torch.from_numpy(y_seg)
        return x, x_seg,y,y_seg

    def __len__(self):
        return len(self.paths)

'''
class JHUBrainInferDataset(Dataset):#连续来使用测试集，是12,23,34,不是12,34,56
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        #print(self.paths)
        path = self.paths[index]
        len_files = len( self.paths )
        if index==len_files-1:
            index1 = 0
        else :
            index1 = index + 1

        first = pkload(self.paths[index])
        x = first[0]
        x_seg = first[1]
        second = pkload(self.paths[index1])
        y = second[0]
        y_seg = second[1]

        x, y = x[None, ...], y[None, ...]
        #x = x[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]

        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x,  x_seg = torch.from_numpy(x),  torch.from_numpy(x_seg)
        y,  y_seg = torch.from_numpy(y),  torch.from_numpy(y_seg)
        return x, x_seg,y,y_seg

    def __len__(self):
        return len(self.paths)
