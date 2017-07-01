# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

from .vgg16 import vgg16
from .vgg16_convs import vgg16_convs
from .vgg16_gan import vgg16_gan
from .vgg16_flow import vgg16_flow
from .vgg16_flow_features import vgg16_flow_features

from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/vgg*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

from .dcgan import dcgan
from .resnet50 import resnet50
from .fcn8_vgg import fcn8_vgg
from . import factory
