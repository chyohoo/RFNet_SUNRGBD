import torch

from models.rfnet import RFNet
from models.resnet.resnet_single_scale_single_attention import *


def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print('{}not in model_state'.format(name))
            continue
        else:
            own_state[name].copy_(param)

    return model

resnet_ = resnet18(pretrained=True, efficient=False, use_bn= True)
model_ = RFNet(resnet_,num_classes=37, use_bn= True)
# model_ = torch.nn.DataParallel(model_)   #并不支持trace nn.
model_ =model_.cuda()

device_ = torch.device('cpu')
new_state_dict_ = torch.load('./model_best.pth',map_location=device_)



model_ = load_my_state_dict(model_, new_state_dict_['state_dict'])


model_.eval()

# image_shape torch.Size([8, 3, 480, 640]) b x c x h x w
# depth_shape torch.Size([8, 1, 480, 640])
# 记得归一化图像 /255

image_example = torch.rand(1,3,480,640).cuda()
depth_example = torch.rand(1,1,480,640).cuda() #保存时的模型使用cuda输入 此处也必须为cuda

with torch.no_grad():
    traced_model = torch.jit.trace(model_,(image_example,depth_example))
    traced_model.save('./traced_model_cpu.pt')
    print('完成')







