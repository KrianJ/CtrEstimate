#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/7 15:50
# software: PyCharm-FM_torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class FM_Module(nn.Module):
    def __init__(self, k, w_reg, v_reg):
        super(FM_Module, self).__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

        self.w0 = torch.tensor(0., requires_grad=True)
