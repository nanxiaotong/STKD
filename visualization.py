import matplotlib.pyplot as plt
from PIL import Image
import requests
import numpy as np
from io import BytesIO
import torch
from torch import nn
from torchvision.models import resnet34
from torchvision.models.resnet import ResNet, BasicBlock
import torchvision.transforms as T
import torch.nn.functional as F
import pylab

# 加载torchvision.models中内置的预训练模型resnet34
base_resnet34 = resnet34(pretrained=True)
# print(base_resnet34)

class ResNet34AT(ResNet):
    """Attention maps of ResNet-34.
    Overloaded ResNet model to return attention maps.
    """    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        g0 = self.layer1(x)#第一个残差块
        g1 = self.layer2(g0)#第二个残差块
        g2 = self.layer3(g1)#第三个残差块
        g3 = self.layer4(g2)#第四个残差块
        
        return [g.pow(2).mean(1) for g in (g0, g1, g2, g3)]
    
model = ResNet34AT(BasicBlock, [3, 4, 6, 3])
# print(model)
model.load_state_dict(base_resnet34.state_dict())

def load(url):
    response = requests.get(url)
    return np.ascontiguousarray(Image.open(BytesIO(response.content)), dtype=np.uint8)

im = load('http://www.zooclub.ru/attach/26000/26132.jpg')
plt.imshow(im)
pylab.show()


#定义对图像的预处理
tr_center_crop = T.Compose([
        T.ToPILImage(),#转化为PILImage
        T.Resize(256),#调整大小为256
        T.ToTensor(),#转化为tensor格式
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#归一化
    ])


model.eval()
with torch.no_grad():
    x = tr_center_crop(im).unsqueeze(0)
    gs = model(x)

for i, g in enumerate(gs):
    plt.imshow(g[0], interpolation='bicubic')#插值运算
    plt.title(f'g{i}')
    plt.show()