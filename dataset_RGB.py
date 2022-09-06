import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from pdb import set_trace as stx
import random
import sys

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'bmp'])

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'resized_256_blur')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'resized_256_original')))

        self.inp_filenames = [os.path.join(rgb_dir, 'resized_256_blur', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'resized_256_original', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert("RGB")
        tar_img = Image.open(tar_path).convert("RGB")

        aug    = random.randint(0, 3)
        if aug == 1:
            # add
            gamma_value = round(random.uniform(0.8, 1.2), 2)
            inp_img = TF.adjust_gamma(inp_img, gamma_value)
            tar_img = TF.adjust_gamma(tar_img, gamma_value)

        if aug == 2:
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            inp_img = TF.adjust_saturation(inp_img, sat_factor)
            tar_img = TF.adjust_saturation(tar_img, sat_factor)
        
        if aug == 3:
            contrast_factor = random.randint(0, 2)
            inp_img = TF.adjust_contrast(inp_img, contrast_factor)
            tar_img = TF.adjust_contrast(tar_img, contrast_factor)


        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]


        return inp_img, tar_img, filename

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'resized_256_blur')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'resized_256_original')))
        self.inp_filenames = [os.path.join(rgb_dir, 'resized_256_blur', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'resized_256_original', x) for x in tar_files if is_image_file(x)]


        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert("RGB")
        tar_img = Image.open(tar_path).convert("RGB")

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, filename

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp).convert("RGB")

        inp = TF.to_tensor(inp)
        return inp, filename
