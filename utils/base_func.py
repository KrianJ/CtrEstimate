#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/7 14:01
# software: PyCharm-base_func

import numpy as np


def sigmoid(x):
    """sigmoid函数"""
    return 1 / (1 + np.exp(-x))
