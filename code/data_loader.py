import os
import json
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision.transforms as transforms

import utils


def side_to_bound(side):
    if side == 'cacheline':
        v_max = 0xFFFF_FFFF >> 6
        v_min = -(0xFFFF_FFFF >> 6)
    elif side == 'pagetable':
        v_max = 0xFFFF_FFFF >> 12
        v_min = -(0xFFFF_FFFF >> 12)
    elif side == 'cachebank':
        v_max = 0xFFFF_FFFF >> 2
        v_min = -(0xFFFF_FFFF >> 2)
    else:
        raise NotImplementedError
    return v_max, v_min

class CelebaDataset(Dataset):
    def __init__(self, npz_dir, img_dir, ID_path, split,
                image_size, side, trace_c, trace_w,
                op=None, k=None):
        super().__init__()
        self.npz_dir = ('%s%s/' % (npz_dir, split))
        self.img_dir = ('%s%s/' % (img_dir, split))
        self.trace_c = trace_c
        self.trace_w = trace_w
        self.op = op
        self.k = k

        self.npz_list = sorted(os.listdir(self.npz_dir))
        self.img_list = sorted(os.listdir(self.img_dir))

        self.transform = transforms.Compose([
                       transforms.Resize(image_size),
                       transforms.CenterCrop(image_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])

        self.v_max, self.v_min = side_to_bound(side)
        
        print('Total %d Data Points.' % len(self.npz_list))

        with open(ID_path, 'r') as f:
            self.ID_dict = json.load(f)

        self.ID_cnt = len(set(self.ID_dict.values()))
        print('Total %d ID.' % self.ID_cnt)

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        npz_name = self.npz_list[index]
        prefix = npz_name.split('.')[0]
        # prefix = int('-'.join(npz_name.split('-')[1:]).split('.')[0])    
        suffix = '.jpg'
        img_name = prefix + suffix
        ID = int(self.ID_dict[img_name]) - 1

        npz = np.load(self.npz_dir + npz_name)
        trace = npz['arr_0']
        trace = trace.astype(np.float32)

        if self.op == 'shift':
            trace = np.concatenate([trace[self.k:], trace[:self.k]])

        if self.op == 'delete':
            assert self.k < 1
            length = len(trace)
            del_num = int(length * self.k)
            del_index = np.random.choice(np.arange(length), del_num, replace=False)
            del_trace = np.delete(trace, del_index)
            trace = np.concatenate([del_trace, np.array([0] * del_num)])
            trace = trace.astype(np.float32)

        trace = torch.from_numpy(trace).view([self.trace_c, self.trace_w, self.trace_w])
        trace = utils.my_scale(v=trace,
                               v_max=self.v_max,
                               v_min=self.v_min
                            )
        if self.op == 'noise':
            trace = (1 - self.k) * trace + self.k * torch.randn(trace.size())

        if self.op == 'zero':
            trace = F.dropout(trace, p=self.k)

        image = Image.open(self.img_dir + img_name)
        image = self.transform(image)

        ID = torch.LongTensor([ID]).squeeze()
        return trace, image, prefix, ID
    
class ImageDataset(Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.img_dir = args.image_dir + ('train/' if split == 'train' else 'test/')

        self.img_list = sorted(os.listdir(self.img_dir))

        self.transform = transforms.Compose([
                       transforms.Resize(args.image_size),
                       transforms.CenterCrop(args.image_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])
        
        print('Total %d Data Points.' % len(self.img_list))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        image = Image.open(self.img_dir + img_name)
        image = self.transform(image)
        
        return image

class DataLoader:
    def __init__(self, args):
        self.args = args
        self.init_param()

    def init_param(self):
        self.gpus = torch.cuda.device_count()
        self.gpus = max(1, self.gpus)

    def get_loader(self, dataset, shuffle=True):
        data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.args.batch_size * self.gpus,
                            num_workers=int(self.args.num_workers),
                            shuffle=shuffle
                        )
        return data_loader
