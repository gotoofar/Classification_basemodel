from torch.autograd import Variable
import torch as t
from torchnet import meter
import time
import torch.optim as optim
from torch.optim import lr_scheduler
from resnet34_50 import resnet50
from dataset import VGGDataset
from torch.utils.data import  DataLoader
import torch.nn as nn
from visdom import Visdom

#模型保存的路径
model_save_path="G:/jupyter_proj/Alexnet/from_1080_class/model_save/"

#选哪一个网络
net = resnet50(pretrained=False)


#net.load_state_dict(t.load("D:/notebook/VGG_relic/model_34_50/relic50_last.pkl"))


if t.cuda.is_available():
    net=net.cuda()
    print('ok')

img_data = VGGDataset(train=True)
data_loader = DataLoader(img_data, batch_size=4, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)


#可视化   python -m visdom.server  启动
viz = Visdom()
viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
viz.line([0.], [0], win='accuracy', opts=dict(title='accuracy'))
ii=0
for epo in range(6):

    scheduler.step()
    running_loss = 0.0

    index = 0
    epo_loss = 0
    print("another epo")

    test_data = VGGDataset(train=False)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=True)
    testiter = iter(test_loader)

    time_start = time.time()
    for i, data in enumerate(data_loader, 0):
        net.train()

        # adjust_learning_rate(optimizer,epo,base_lr=0.001)

        inputs, y, _ = data  # 这里加了一个 名字的 在训练里可以省略
        index += 1
        # print(inputs.shape)
        inputs = t.autograd.Variable(inputs)
        y = t.autograd.Variable(y)

        inputs = inputs.cuda()
        y = y.cuda()

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #多少代出一次loss
        if i % 20 == 19:
            print(epo + 1, i + 1, running_loss / 20)
            ii+=1
            viz.line([running_loss / 20], [ii], win='train_loss', update='append')
            running_loss = 0.0
        #多少代测试一次精度
        if i % 100 == 99:
            images_show, labels_show, _ = testiter.next()
            correct_val = 0
            total_val = 0
            images_show = images_show.cuda()
            labels_show = labels_show.cuda()
            net.eval()
            outputs_show = net(Variable(images_show))
            _, predicted = t.max(outputs_show.data, 1)

            total_val += labels_show.size(0)
            correct_val += (predicted == labels_show).sum().cpu().numpy()

            print("acc_test=", correct_val / total_val)
            viz.line([correct_val / total_val], [ii], win='accuracy', update='append')

    #多少代保存一次模型
    if epo % 2 == 1:
        t.save(net.state_dict(), model_save_path+'test50_' + str(epo) + '.pkl')

    time_end = time.time()
    print(time_end - time_start)
# t.save(net.state_dict(), 'model_34_50/relic50_last.pkl')
# t.save(net, 'model_34_50/relic50_last.pth')

print('Finished')