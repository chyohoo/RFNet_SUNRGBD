import torch
import torch.nn as nn
from itertools import chain # 串联多个迭代对象

from .util import _BNReluConv, upsample


class RFNet(nn.Module):
    # __constants__ = ['logits']
    def __init__(self, backbone, num_classes, use_bn=True):
        super(RFNet, self).__init__()
        self.  backbone = backbone
        self.num_classes = num_classes
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)

    # @torch.jit.script_method
    def forward(self, rgb_inputs, depth_inputs = None):
        # print("rgb_rf_inputs",rgb_inputs.shape)
        # print("depth_rf_shape",depth_inputs.shape)
        x, additional = self.backbone(rgb_inputs, depth_inputs)
        # print("x_shape",x.shape)
        logits = self.logits.forward(x)
        # print("logit_fwd",logits)
        # print("logit_fwd_shape", logits.shape)
        logits_upsample = upsample(logits, rgb_inputs.shape[2:])
        return logits_upsample

    # @torch.jit.ignore
    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))
    
    # @torch.jit.ignore
    def fine_tune_params(self):
        return self.backbone.fine_tune_params()
