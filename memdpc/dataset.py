import torch
from torch.utils import data
from torchvision import transforms
import os
import sys
import time
import pickle
import glob
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

def read_file(path):
    with open(path, 'r') as f:
        content = f.readlines()
    content = [i.strip() for i in content]
    return content

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class K400Dataset(data.Dataset):
    def __init__(self,
                 root='%s/../process_data/data/k400' % os.path.dirname(os.path.abspath(__file__)),
                 mode='val',
                 transform=None,
                 seq_len=5,
                 num_seq=8,
                 downsample=3,
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.return_label = return_label

        classes = read_file(os.path.join(root, 'ClassInd.txt'))
        print('Frame Dataset from {} has #class {}'.format(root, len(classes)))
        self.num_class = len(classes)
        self.class_to_idx = {classes[i]:i for i in range(len(classes))}
        self.idx_to_class = {i:classes[i] for i in range(len(classes))}

        # splits
        if mode == 'train':
            split = '../process_data/data/kinetics400/train_split.csv'
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = '../process_data/data/kinetics400/val_split.csv'
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx) 
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': 
            self.video_info = self.video_info.sample(frac=0.3, random_state=666)

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        start_idx = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), 1)
        seq_idx = np.arange(self.num_seq*self.seq_len)*self.downsample + start_idx
        return seq_idx

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        frame_index = self.idx_sampler(vlen, vpath)
        
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in frame_index]
        t_seq = self.transform(seq) 
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)

            label = torch.LongTensor([vid])
            return t_seq, label

        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        return self.class_to_idx[action_name]

    def decode_action(self, action_code):
        return self.idx_to_class[action_code]


class UCF101Dataset(data.Dataset):
    def __init__(self,
                 root='%s/../process_data/data/ucf101' % os.path.dirname(os.path.abspath(__file__)),
                 mode='val',
                 transform=None, 
                 seq_len=5,
                 num_seq=8,
                 downsample=3,
                 which_split=1,
                 return_label=False,
                 return_path=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.which_split = which_split
        self.return_label = return_label
        self.return_path = return_path

        # splits
        if mode == 'train':
            split = '../process_data/data/ucf101/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'): # use val for test
            split = '../process_data/data/ucf101/test_split%02d.csv' % self.which_split 
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        classes = read_file(os.path.join(root, 'ClassInd.txt'))
        print('Frame Dataset from {} has #class {}'.format(root, len(classes)))
        self.num_class = len(classes)
        self.class_to_idx = {classes[i]:i for i in range(len(classes))}
        self.idx_to_class = {i:classes[i] for i in range(len(classes))}

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': 
            self.video_info = self.video_info.sample(frac=0.3)

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if self.mode == 'test':
            available = vlen-self.num_seq*self.seq_len*self.downsample
            start_idx = np.expand_dims(np.arange(0, available+1, self.num_seq*self.seq_len*self.downsample//2-1), 1)
            seq_idx = np.expand_dims(np.arange(self.num_seq*self.seq_len)*self.downsample, 0) + start_idx # [test_sample, num_frames]
            seq_idx = seq_idx.flatten(0)
        else:
            start_idx = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), 1)
            seq_idx = np.arange(self.num_seq*self.seq_len)*self.downsample + start_idx
        return seq_idx


    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        frame_index = self.idx_sampler(vlen, vpath)
        
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in frame_index]
        t_seq = self.transform(seq)
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        if self.mode == 'test':
            t_seq = t_seq.view(-1, self.num_seq, self.seq_len, C, H, W).transpose(2,3)
        else:
            t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            if self.return_path:
                return t_seq, (label, vpath)
            else:
                return t_seq, label
            
        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        return self.class_to_idx[action_name]

    def decode_action(self, action_code):
        return self.idx_to_class[action_code]


class HMDB51Dataset(data.Dataset):
    def __init__(self,
                 root='%s/../process_data/data/hmdb51' % os.path.dirname(os.path.abspath(__file__)),
                 mode='val',
                 transform=None, 
                 seq_len=5,
                 num_seq=8,
                 downsample=3,
                 which_split=1,
                 return_label=False,
                 return_path=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.which_split = which_split
        self.return_label = return_label
        self.return_path = return_path

        # splits
        if mode == 'train':
            split = '../process_data/data/hmdb51/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'): # use val for test
            split = '../process_data/data/hmdb51/test_split%02d.csv' % self.which_split 
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        classes = read_file(os.path.join(root, 'ClassInd.txt'))
        print('Frame Dataset from {} has #class {}'.format(root, len(classes)))
        self.num_class = len(classes)
        self.class_to_idx = {classes[i]:i for i in range(len(classes))}
        self.idx_to_class = {i:classes[i] for i in range(len(classes))}

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': 
            self.video_info = self.video_info.sample(frac=0.3)

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if self.mode == 'test':
            available = vlen-self.num_seq*self.seq_len*self.downsample
            start_idx = np.expand_dims(np.arange(0, available+1, self.num_seq*self.seq_len*self.downsample//2-1), 1)
            seq_idx = np.expand_dims(np.arange(self.num_seq*self.seq_len)*self.downsample, 0) + start_idx # [test_sample, num_frames]
            seq_idx = seq_idx.flatten(0)
        else:
            start_idx = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), 1)
            seq_idx = np.arange(self.num_seq*self.seq_len)*self.downsample + start_idx
        return seq_idx


    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        frame_index = self.idx_sampler(vlen, vpath)
        
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in frame_index]
        t_seq = self.transform(seq)
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        if self.mode == 'test':
            t_seq = t_seq.view(-1, self.num_seq, self.seq_len, C, H, W).transpose(2,3)
        else:
            t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            if self.return_path:
                return t_seq, (label, vpath)
            else:
                return t_seq, label
            
        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        return self.class_to_idx[action_name]

    def decode_action(self, action_code):
        return self.idx_to_class[action_code]



class CATERDataset(data.Dataset):
    def __init__(self,
                 root='/mnt/285EDDF95EDDC02C/Users/Public/Documents/VideoDatasets/CATER/max2actions',
                 mode='val',
                 task='actions_present',
                 transform=None,
                 seq_len=5,
                 num_seq=8,
                 downsample=3,
                 return_label=False,
                 return_path=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.return_label = return_label
        self.return_path = return_path

        # splits
        if mode == 'train':
            path_split = "lists/{}/train_subsetT.txt"
        elif mode == 'val':
            path_split = "lists/{}/train_subsetV.txt"
        elif mode == 'test':
            path_split = "lists/{}/val.txt"
        else: raise ValueError('wrong mode')

        path_split = path_split.format(task)

        paths = []
        labels = []
        black_list = [75, 76, 385, 4798, 4803, 4820, 6531, 6532, 6536]
        with open(os.path.join(root, path_split),"r") as f: 
            for s in f.readlines(): 
                path = s.split(' ')[0].split('.')[0]
                if int(path.split('_')[-1]) in black_list:
                    print("Skipping video {}".format(path))
                    continue
                path = os.path.join(root, 'frame/videos', path)
                paths.append(path)
                labels.append([int(x) for x in s.split(' ')[1].split(',')])
        self.labels = labels
        self.num_class = max([x for l in labels for x in l])+1
        self.paths = paths
        print('Frame Dataset from {} has #class {}'.format(root, self.num_class))

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if self.mode == 'test':
            available = vlen-self.num_seq*self.seq_len*self.downsample
            start_idx = np.expand_dims(np.arange(0, available+1, self.num_seq*self.seq_len*self.downsample//2-1), 1)
            seq_idx = np.expand_dims(np.arange(self.num_seq*self.seq_len)*self.downsample, 0) + start_idx # [test_sample, num_frames]
            seq_idx = seq_idx.flatten()
        else:
            start_idx = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), 1)
            seq_idx = np.arange(self.num_seq*self.seq_len)*self.downsample + start_idx
        return seq_idx

    def __getitem__(self, index):
        vpath = self.paths[index]
        vlen = 300
        frame_index = self.idx_sampler(vlen, vpath)
        
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in frame_index]
        t_seq = self.transform(seq) 
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        if self.mode == 'test':
            t_seq = t_seq.view(-1, self.num_seq, self.seq_len, C, H, W).transpose(2,3)
        else:
            t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        if self.return_label:
            label = torch.LongTensor(self.labels[index])
            label = torch.nn.functional.one_hot(label, self.num_class).sum(0)
            if self.return_path:
                return t_seq, (label, vpath)
            return t_seq, label

        return t_seq

    def __len__(self):
        return len(self.paths)