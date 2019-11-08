# Classification_basemodel
 > Some base model like VGG Resnet in  a classification project

整理了最早使用的几个分类网络 使用时设置好路径和参数 python train.py即可
或者在pycharm里面点run
如果需要可视化loss和accuracy，则启动visdom（python -m visdom.server）
test.py  可用于大批量测试图片得到准确率（TOP1 & TOP3）并找出分类错误图片的名字，具体使用在程序里有