import time # 时间模块， 用于记录每次迭代的时间打印到控制台
import os # 操作系统模块， 用于创建文件夹，以及路径的一些操作
import numpy as np, math, copy
import matplotlib.pyplot as plt
import torch # 导入pytorch核心模块
import torch.nn as nn # nn模块是pytorch和神经网络相关的模块
from torch.utils.data import Dataset # 从torch.utils模块获取数据集的操作的模块
from torchvision import datasets, transforms # 获取datasets和transforms模块，这两个模块主要用于处理数据。
# datasets是加载数据，transforms是将数据进行处理，
# 比如将PILImage图像转换成Tensor，或者将Tensor转换成PILImage图像
import pickle # pickle模块主要是用来将对象序列化存储到硬盘
import PIL.Image as Image # 处理图像数据的模块
from utils.femnist import FEMNIST # 参数命令行交互模块
import argparse

os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
人脸数据集的一些处理
'''
class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab


def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst



'''
自定义LeNet网络
'''
class LeNet(nn.Module): # nn.Module, 定义神经网络必须继承的模块， 框架规定的形式
    def __init__(self, channel=3, hidden=768, num_classes=10): # 假设输入cifar10数据集， 默认3通道， 隐层维度为768， 分类为10
        super(LeNet, self).__init__() # 继承pytorch神经网络工具箱中的模块
        act = nn.Sigmoid # 激活函数为Sigmoid
        # nn.Sequential: 顺序容器。 模块将按照在构造函数中传递的顺序添加到模块中。 或者，也可以传递模块的有序字典
        self.body = nn.Sequential( # 设计神经网络结构，对于nn.Sequential.Preference : https://zhuanlan.zhihu.com/p/75206669
            # 设计输入通道为channel，输出通道为12， 5x5卷积核尺寸，填充为5 // 2是整除。故填充为2， 步长为2的卷积层
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            # 经过卷积后， 使用Sigmoid激活函数激活
            act(),
            # 设计输入通道为12，输出通道为12， 5x5卷积核尺寸，填充为5 // 2是整除。故填充为2， 步长为2的卷积层
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            # 经过卷积后， 使用Sigmoid激活函数激活
            act(),
            # 设计输入通道为12，输出通道为12， 5x5卷积核尺寸，填充为5 // 2是整除。故填充为2， 步长为1的卷积层
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            # 经过卷积后， 使用Sigmoid激活函数激活
            act()
        )
        # 设计一个全连接映射层， 将hidden隐藏层映射到十个分类标签
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    # 设计前向传播算法
    def forward(self, x):
        out = self.body(x) # 先经过nn.Sequential的顺序层得到一个输出
        out = out.view(out.size(0), -1) # 将输出转换对应的维度
        out = self.fc(out) # 最后将输出映射到一个十分类的一个列向量
        return out

'''
init weights
'''
def weights_init(m):
    try:
        if hasattr(m, "weight"): # hasattr：函数用于判断对象是否包含对应的属性。
            m.weight.data.uniform_(-0.5, 0.5) # 对m.weight.data进行均值初始化。m.weights.data指的是网络中的卷积核的权重
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"): # 对偏置进行初始化
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


seed = 1234 # 经过专家的实验， 随机种子数为1234结果会较好
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num = 100
sigma = 0.001
label = 1

dataset = "cifar10"
root_path = 'data/'
data_path = os.path.join(root_path, dataset)
save_path = os.path.join('results/DLG_%s' % dataset).replace('\\', '/')  # 图片保存的路径

lr = 1.0 # 学习率
iteration = 100 # 一张图片迭代的次数
num_exp = 1 # 实验次数也就是神经网络训练的epoch
use_cuda = torch.cuda.is_available() # 是否可以使用gpu，返回值为True或False
device = 'cuda' if use_cuda else 'cpu' # 设置 device是cpu或者cuda

tt = transforms.Compose([transforms.ToTensor()]) # 将图像类型数据（PILImage）转换成Tensor张量
tp = transforms.Compose([transforms.ToPILImage()]) # 将Tensor张量转换成图像类型数据

if not os.path.exists('results'): # 判断是否存在results文件夹，没有就创建，Linux中mkdir创建文件夹
    os.mkdir('results')
if not os.path.exists(save_path): # 是否存在路径， 不存在则创建保存图片的路径
    os.mkdir(save_path)

if dataset == 'MNIST' or dataset == 'mnist':  # 判断是什么数据集
    image_shape = (28, 28)  # mnist数据集图片尺寸是28x28
    num_classes = 10  # mnist数据分类为十分类： 0 ～ 9
    channel = 1  # mnist数据集是灰度图像所以是单通道
    hidden = 588  # hidden是神经网络最后一层全连接层的维度
    dst = datasets.MNIST(data_path, download=True)

elif dataset == 'cifar10' or dataset == 'CIFAR10':
    image_shape = (32, 32)  # cifar10数据集图片尺寸是32x32
    num_classes = 10  # cifar10数据分类为十分类：卡车、 飞机等
    channel = 3  # cifar10数据集是RGB图像所以是三通道
    hidden = 768  # hidden是神经网络最后一层全连接层的维度
    dst = datasets.CIFAR10(data_path, download=True)

elif dataset == 'femnist':
    image_shape = (28, 28)  # cifar10数据集图片尺寸是32x32
    num_classes = 10  # cifar10数据分类为十分类：卡车、 飞机等
    channel = 1  # cifar10数据集是RGB图像所以是三通道
    hidden = 588  # hidden是神经网络最后一层全连接层的维度
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_classes = 10
    dst = FEMNIST(args, data_path, train=True, download=True)

nets = []
for i in range(num):
    net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes).to(device) # 初始化LeNet模型
    net.apply(weights_init) # 初始化模型中的卷积核的权重
    nets.append(net)

criterion = nn.CrossEntropyLoss().to(device) # 设置损失函数为交叉熵函数

index_list = []
gt_data_list = []
gt_label_list = []

for i in range(50000):
    if dst[i][1]==label:
        index_list.append(i)
        if len(index_list)==num:
            break

for i in range(len(index_list)):
    index = index_list[i]
    tmp_datum = tt(dst[index][0]).float().to(device) # 将数据集中index对应的图片数据拿出来转换成Tensor张量
    tmp_datum = tmp_datum.view(1, *tmp_datum.size()) # 将tmp_datum数据重构形状， 可以用shape打印出来看看
    tmp_label = torch.Tensor([dst[index][1]]).long().to(device) # 将数据集中index对应的图片的标签拿出来转换成Tensor张量
    tmp_label = tmp_label.view(1, ) # 将标签重塑为列向量形式
    gt_data_list.append(tmp_datum)
    gt_label_list.append(tmp_label)

# compute original gradient
for i in range(num):
    out = nets[i](gt_data_list[i]) # 将真实图片数据丢入到net网络中获得一个预测的输出
    y = criterion(out, gt_label_list[i]) # 使用交叉熵误差函数计算真实数据的预测输出和真实标签的误差
    dy_dx = torch.autograd.grad(y, nets[i].parameters()) # 通过自动求微分得到真实梯度
    if i==0:
        original_dy_dx = list((_.detach().clone()/num for _ in dy_dx))
    else:
        for o, _ in zip(original_dy_dx, dy_dx):
            o.data += _.data/num

S=[]
for i in range(len(original_dy_dx)):
    S.append(torch.norm(original_dy_dx[i], 2))
S_value = np.median(S)
for i, o in enumerate(original_dy_dx):
    noise = torch.normal(0, S_value * sigma, size = o.shape)/float(num)   
    o.data += noise.data

onet = copy.deepcopy(nets[0])
# generate dummy data and label。 生成假的数据和标签
dummy_data = torch.randn(gt_data_list[0].size()).to(device).requires_grad_(True)
dummy_label = torch.randn((gt_label_list[0].shape[0], num_classes)).to(device).requires_grad_(True)

optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr) #设置优化器为拟牛顿法

history = [] # 记录全部的假的数据（这里假的数据指的是随机产生的假图像）
history_iters = [] # 记录画图使用的迭代次数

for iters in range(iteration): # 开始训练迭代

    def closure(): # 闭包函数
        optimizer.zero_grad() # 每次都将梯度清零
        pred = onet(dummy_data) # 将假的图片数据丢给神经网络求出预测的标签

        # 将假的预测进行softmax归一化，转换为概率
        dummy_loss = -torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))

        # 对假的数据进行自动微分， 求出假的梯度
        dummy_dy_dx = torch.autograd.grad(dummy_loss, onet.parameters(), create_graph=True)

        grad_diff = 0 # 定义真实梯度和假梯度的差值
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): # 对应论文中的假的梯度减掉真的梯度平方的式子
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward() # 对||dw假 - dw真|| 进行反向传播进行更新
        return grad_diff

    optimizer.step(closure) # 优化器更新梯度

max_psnr = 0
for jj in range(num):
    img1 = gt_data_list[jj][0]
    img2 = dummy_data[0]
    psnr = 0.0
    for td in range(channel):
        td1 = np.array(img1[td].detach().numpy()).astype(np.float64)
        td2 = np.array(img2[td].detach().numpy()).astype(np.float64)
        mse = np.mean((td1-td2)**2)
        psnr += 20.0*np.log10(255.0/np.sqrt(mse))
    max_psnr = max(max_psnr, psnr/channel)
print(max_psnr)

# plt.xticks([])  # 去掉x轴
# plt.yticks([])  # 去掉y轴
# plt.axis('off')
# # plt.title('origin', fontsize=24)
# plt.title('PSNR=%.2f' % max_psnr, fontsize=24)
# if channel==1:
#     plt.imshow(dummy_data[0][0].detach().numpy(), cmap='gray') # 灰度图像
# else:
#     image = dummy_data[0].detach().numpy().reshape(-1,1024)
#     r = image[0,:].reshape(32,32) #红色分量
#     g = image[1,:].reshape(32,32) #绿色分量
#     b = image[2,:].reshape(32,32) #蓝色分量
#     img = np.zeros((32,32,3))
#     img[:,:,0]=r
#     img[:,:,1]=g
#     img[:,:,2]=b
#     plt.imshow(img) # 彩色图像
# plt.savefig('%s/DLG.png' % (save_path)) # 保存图片地址
# plt.close()
