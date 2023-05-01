import os
from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, args, split):
        super(ImageDataset).__init__()
        self.args = args
        self.img_dir = args['image_dir'] + ('train/' if split == 'train' else 'test/')

        self.img_list = sorted(os.listdir(self.img_dir))

        self.transform = transforms.Compose([
                       transforms.Resize(args['image_size']),
                       transforms.CenterCrop(args['image_size']),
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

    def get_loader(self, dataset):
        data_loader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size=self.args['batch_size'] * self.gpus,
                            num_workers=int(self.args['num_workers']),
                            shuffle=True
                        )
        return data_loader
