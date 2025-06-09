import argparse
import os
import sys
import numpy as np
from math import log10
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from tensorboard_logger import configure, log_value
import torchvision.utils as vutils
from model_gradient import model
#import model_li
from utils import Visualizer
import os
import torch.nn.parallel
from multiprocessing import Process
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as Fun
from torchvision.utils import save_image
import pytorch_ssim
from exclusion_loss import compute_gradient_img,exclusion_loss
from PIL import Image
import time
#GPUID = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#model =nn.DataParallel(models)
parser = argparse.ArgumentParser()

parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size') # param
parser.add_argument('--imageSize', type=int, default=64, help='the low resolution image size')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')
Loss_list = []
PSNR_list = []

opt = parser.parse_args()
root_tt = './dataset/FOGIV/test/test_'

def default_loader(path):
    return Image.open(path)

class Train_Dataset(Dataset):
    def __init__(self, img_root, txt_name, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt_name, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            input = words[0]
            im_root_vi = img_root
            im_root_ir = img_root.replace('vi', 'ir')
            im_root_gt = img_root.replace('vi', 'gt')
            imgs.append([im_root_vi + '/' + input, im_root_ir + '/' + input, im_root_gt + '/' + input])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path_vi, path_ir, path_gt = self.imgs[index]
        words = path_vi.split('./dataset/FOGIV/test/test_vi/')
        img_name = words[1]
        im_vi = self.loader(path_vi)
        im_ir = self.loader(path_ir)
        im_gt = self.loader(path_gt)
        if self.transform is not None:
            im_vi = self.transform(im_vi)
            im_ir = self.transform(im_ir)
            im_gt = self.transform(im_gt)
        return im_vi, im_ir, im_gt, img_name

    def __len__(self):
        return len(self.imgs)

train_test_data = Train_Dataset(img_root=root_tt+'vi', txt_name=root_tt + 'vi.txt', transform=transforms.Compose([transforms.ToTensor()]))
assert Train_Dataset
dataloader_train_test = DataLoader(dataset=train_test_data, batch_size=1, shuffle=False, num_workers=int(opt.workers))
def _load_ckpt(model, ckpt):
    load_dict = {}
    for name in ckpt:
        print(name.split('module.')[1])
        load_dict[name.split('module.')[1]]= ckpt[name]
    model.load_state_dict(load_dict, strict=False)
    return model

generator = model(4)
path = ('./models/FOGIV.pth')
generator = _load_ckpt(generator, torch.load(path))
def params_count(model):

  return np.sum([p.numel() for p in model.parameters()]).item()


print("Params(M): %.3f" % (params_count(generator) / (1000 ** 2)))

if opt.cuda:
    generator.cuda()
total_time = 0.0
total_images = 0
for j, data_tt in enumerate(dataloader_train_test):
    im_vi_t = data_tt[0]
    im_ir_t = data_tt[1]
    im_gt_t = data_tt[2]
    im_name = data_tt[3]
    if opt.cuda:
        im_vi_t = Variable(im_vi_t.cuda())#data_res = torch.nn.DataParallel(Variable(data_res),device_ids=[0,1])
        im_ir_t = Variable(im_ir_t.cuda())#data_real = torch.nn.DataParallel(Variable(data_real),device_ids=[0,1])
        im_gt_t = Variable(im_gt_t.cuda())
        im_mg_t = torch.cat((im_vi_t,im_ir_t),1)
        torch.cuda.synchronize()
        start_time = time.time()
        t1, t2 = generator(im_mg_t)
        torch.cuda.synchronize()
        end_time = time.time()
    else:
        im_vi_t = Variable(im_vi_t)#data_res = torch.nn.DataParallel(Variable(data_res),device_ids=[0,1])
        im_ir_t = Variable(im_ir_t)#data_real = torch.nn.DataParallel(Variable(data_real),device_ids=[0,1]
        im_mg_t = torch.cat((im_vi_t,im_ir_t),1)
        t1,t2 = generator(im_mg_t)

    elapsed_time = end_time - start_time
    total_time += elapsed_time
    total_images += 1

    t0 = (t1 + t2) / 2
    im_vi_gn = torch.mul(t0, im_vi_t)
    im_ir_gn = torch.mul(1-t0, im_ir_t)
    im_mg_gn_t = im_vi_gn + im_ir_gn

    cvt_im_gn = torchvision.transforms.functional.to_pil_image(im_mg_gn_t.squeeze(0).detach().cpu())
    print(type(cvt_im_gn))
    outputtest = './output/FOGIV/'
    try:
        os.makedirs(outputtest)
    except OSError:
        pass

    cvt_im_gn.save(outputtest +str(im_name)+'_FIVFusion.png')

print(f"\nTotal processing time: {total_time:.4f} seconds")
print(f"Number of images processed: {total_images}")
print(f"Average time per image: {total_time/total_images:.4f} seconds")
