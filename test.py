# -*- coding=utf-8 -*-


from torchvision.transforms import ToPILImage
import torchvision as tv
from torch.autograd import Variable
import numpy as np
import heapq
import time
import torch as t
from dataset import VGGDataset
from resnet34_50 import resnet50
from torch.utils.data import  DataLoader

'''
写一个测试代码  希望可以跑完测试集得到一个总体的精度 以判断模型优劣
测试图片路径是直接从dataset里面设置的 
可以得到的结果是 TOP1 准确率 和 TOP3准确率 以及分类错误的图像全路径名称
用上了cuda
'''

net = resnet50(pretrained=False)


net.load_state_dict(t.load("G:/jupyter_proj/Alexnet/from_1080_class/model_save/test50_1.pkl"))


if t.cuda.is_available():
    net=net.cuda()



net.eval()
show = ToPILImage()
Batch_size = 8
test_data = VGGDataset(train=False)
test_loader = DataLoader(test_data, batch_size=Batch_size, shuffle=False)

accu_top3 = 0
accu_top1 = 0

list_wrong = []  #保存
time_start = time.time()

total = 0
correct_1 = 0
correct_3 = 0

leaky=0

for i, data in enumerate(test_loader, 0):
    images_show, labels_show, img_name = data    # [8,3,224,224]  [8]  tuple(8)
    images_show = images_show.cuda()

    labels_show = labels_show.cuda()


    outputs_show = net(Variable(images_show))
    _, predicted = t.max(outputs_show.data, 1)


    outputs_show = outputs_show.cpu().data.numpy()


    for each_th in range(Batch_size):
        if (images_show.size()[0] < Batch_size):
            leaky=images_show.size()[0]
            break
        top3 = heapq.nlargest(3, range(len(outputs_show[each_th])), outputs_show[each_th].take)   #第三个参数是key 这里把索引给他
        if (labels_show[each_th] in top3):
            correct_3 += 1
        else :
            print(labels_show[each_th])
            print(top3)

    total += labels_show.size(0)
    correct_1 += (predicted == labels_show).sum().cpu().numpy()


    for j in range(images_show.size()[0]):

        if (predicted[j] != labels_show[j]):
            list_wrong.append(img_name[j])



time_end = time.time()

print("cost time "+ str(time_end - time_start))

accu_top3 += (correct_3 / (total-leaky))
print("top3 accuracy:")
print(accu_top3)

print("top1 accuracy:")

accu_top1 += (correct_1 / total)
print(accu_top1)

print("mission complete, if you want get the wrong image,use the code below")

#for i in range(len(list_wrong)):
 #   print(list_wrong[i])
