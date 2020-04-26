import numpy as np
import scipy.io
import imageio
# from PIL import Image
import h5py
import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from dataloaders import custom_transforms as tr
from mypath import Path

image_h = 480
image_w = 640


img_dir_train_file = './data/img_dir_train.txt'
depth_dir_train_file = './data/depth_dir_train.txt'
label_dir_train_file = './data/label_train.txt'
img_dir_test_file = './data/img_dir_test.txt'
depth_dir_test_file = './data/depth_dir_test.txt'
label_dir_test_file = './data/label_test.txt'


class SUNRGBD(Dataset):
    def __init__(self, args, phase="train", data_dir=Path.db_root_dir('SUNRGBD')):

        self.phase = phase 
        self.args = args
        self.data_dir = data_dir


        # self.transform = transform

        try:
            with open(img_dir_train_file, 'r') as f:
                self.img_dir_train = f.read().splitlines()
            with open(depth_dir_train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            with open(label_dir_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
            with open(img_dir_test_file, 'r') as f:
                self.img_dir_test = f.read().splitlines()
            with open(depth_dir_test_file, 'r') as f:
                self.depth_dir_test = f.read().splitlines()
            with open(label_dir_test_file, 'r') as f:
                self.label_dir_test = f.read().splitlines()
        except:
            SUNRGBDMeta_dir = os.path.join(self.data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')
            allsplit_dir = os.path.join(self.data_dir, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
            SUNRGBD2Dseg_dir = os.path.join(self.data_dir, 'SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat')
            
            self.img_dir_train = []
            self.depth_dir_train = []
            self.label_dir_train = []
            self.img_dir_test = []
            self.depth_dir_test = []
            self.label_dir_test = []
            self.SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')

            SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                                           struct_as_record=False)['SUNRGBDMeta']
            split = scipy.io.loadmat(allsplit_dir, squeeze_me=True, struct_as_record=False)
            split_train = split['alltrain']

            seglabel = self.SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

            for i, meta in enumerate(SUNRGBDMeta):
                meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
                real_dir = meta_dir.replace('/n/fs/sun3d/data', self.data_dir)
                depth_bfx_path = os.path.join(real_dir, 'depth_bfx/' + meta.depthname)
                rgb_path = os.path.join(real_dir, 'image/' + meta.rgbname)

                label_path = os.path.join(real_dir, 'label/label.npy')

                if not os.path.exists(label_path):
                    os.makedirs(os.path.join(real_dir, 'label'), exist_ok=True)
                    label = np.array(self.SUNRGBD2Dseg[seglabel.value[i][0]].value.transpose(1, 0))
                    np.save(label_path, label)

                if meta_dir in split_train:
                    self.img_dir_train = np.append(self.img_dir_train, rgb_path)
                    self.depth_dir_train = np.append(self.depth_dir_train, depth_bfx_path)
                    self.label_dir_train = np.append(self.label_dir_train, label_path)
                else:
                    self.img_dir_test = np.append(self.img_dir_test, rgb_path)
                    self.depth_dir_test = np.append(self.depth_dir_test, depth_bfx_path)
                    self.label_dir_test = np.append(self.label_dir_test, label_path)

            local_file_dir = '/'.join(img_dir_train_file.split('/')[:-1])
            if not os.path.exists(local_file_dir):
                os.mkdir(local_file_dir)
            with open(img_dir_train_file, 'w') as f:
                f.write('\n'.join(self.img_dir_train))
            with open(depth_dir_train_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_train))
            with open(label_dir_train_file, 'w') as f:
                f.write('\n'.join(self.label_dir_train))
            with open(img_dir_test_file, 'w') as f:
                f.write('\n'.join(self.img_dir_test))
            with open(depth_dir_test_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_test))
            with open(label_dir_test_file, 'w') as f:
                f.write('\n'.join(self.label_dir_test))

    def __len__(self):
        if self.phase == 'train':
            return len(self.img_dir_train)
        elif self.phase == 'test':
            return len(self.img_dir_test[:-2000])
        elif self.phase == 'val':
            return len(self.img_dir_test[-2000:])

    def __getitem__(self, idx):
        if self.phase == 'train':
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
        elif self.phase == 'test':
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test
            label_dir = self.label_dir_test
        elif self.phase == 'val':
            img_dir = self.img_dir_train[-2000:]
            depth_dir = self.depth_dir_train[-2000:]
            label_dir = self.label_dir_train[-2000:]

        label = np.load(label_dir[idx])
        depth = imageio.imread(depth_dir[idx])
        image = imageio.imread(img_dir[idx])
        

        sample = {'image': image, 'depth': depth, 'label': label}

        if self.phase == "train":
            # sam = self.transform_tr(sample)
            # print(sam['image'])
            return self.transform_tr(sample)
        elif self.phase == "test":
            return self.transform_ts(sample)
        elif self.phase == "val":
            return self.transform_val(sample), img_dir


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.scaleNorm(),
            tr.RandomScale((1.0,1.4)),
            tr.RandomHSV((0.9, 1.1),
                         (0.9, 1.1),
                         (25, 25)),
            tr.RandomCrop(image_h,image_w),
            tr.RandomFlip(),
            tr.ToTensor(),
            tr.Normalize()])

        return composed_transforms(sample)


    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.scaleNorm(),
            tr.ToTensor(),
            tr.Normalize()])

        return composed_transforms(sample)


    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            
            tr.scaleNorm(),
            tr.ToTensor(),
            tr.Normalize()])

        return composed_transforms(sample)









