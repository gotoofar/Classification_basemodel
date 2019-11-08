# -*- coding=utf-8 -*-
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from torch.utils import data
from PIL import Image
from torchvision import transforms
from torchvision import datasets

from torch.utils.data import Dataset

'''
数据载入代码
下面的路径分别填上训练集位置和测试集位置
存放格式都是每类图片一个文件夹，名字为类名
'''


transform_data = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                          std=[0.5, 0.5, 0.5])])
transform_data_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                               std=[0.5, 0.5, 0.5])])


class VGGDataset(Dataset):

    def __init__(self, train=True):
        if train:
            self.img_datas = datasets.ImageFolder('G:/jupyter_proj/Alexnet/from_1080_class/data_train/')
            self.transform = transform_data
        else:
            self.img_val = datasets.ImageFolder('G:/jupyter_proj/Alexnet/from_1080_class/data_test/')
            self.transform = transform_data_test

        self.train = train

    def __getitem__(self, idx):

        if self.train:
            # print("train")
            imgA = cv2.imread(self.img_datas.imgs[idx][0])
            imgA = cv2.resize(imgA, (224, 224))
            label = self.img_datas.imgs[idx][1]
            imgA = self.transform(imgA)
            img_name = " "


        else:
            # print("Test")
            try:
                imgA = cv2.imread(self.img_val.imgs[idx][0])
                imgA = cv2.resize(imgA, (224, 224))
                label = self.img_val.imgs[idx][1]
                imgA = self.transform(imgA)
                img_name = self.img_val.imgs[idx][0]
            except:
                print(self.img_val.imgs[idx][0])

        return imgA, label, img_name

    def __len__(self):
        if self.train:
            return len(self.img_datas.imgs)
        else:
            return len(self.img_val.imgs)