from io import BytesIO
import os
import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torch


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            print(f'{self.resolution}-{str(index).zfill(5)}')
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            # print(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        # print('after open buffer in dataset')
        img = self.transform(img)

        return img

class ImageDataset(Dataset):
    def __init__(self, path, transform, resolution=256, to_crop=False):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

class ProjectorDataset(Dataset):
    def __init__(self, abs_dir, size, size_ratio):


        resize = min(size, 256)
        if size_ratio != 1:
            resize = [resize, resize * 2]
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.listFullPath = []
        if 'cityscapes' in abs_dir:
            print('Preparing Cityscapes Dataset')
            for city_folder in sorted(os.listdir(abs_dir)):
                cur_folder = os.path.join(abs_dir, city_folder)
                for item in sorted(os.listdir(cur_folder)):
                    if os.path.isfile(os.path.join(cur_folder, item)):
                        self.listFullPath.append(os.path.join(cur_folder, item))
        else:
            nam_f = os.listdir(abs_dir)
            for imgfile in nam_f:
                full_path = os.path.join(abs_dir,imgfile)
                if os.path.isfile(full_path):
                    self.listFullPath.append(full_path)

        self.length = len(self.listFullPath)
        print("length of dataset :", self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        fullPathImg = self.listFullPath[index]
        img = self.transform(Image.open(fullPathImg).convert("RGB"))
        return img, fullPathImg
