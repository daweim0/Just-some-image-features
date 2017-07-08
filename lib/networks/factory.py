# --------------------------------------------------------
# FCN
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

# import networks.vgg16
# import networks.vgg16_convs
# import networks.vgg16_gan
# import networks.vgg16_flow
# import networks.dcgan
# import networks.resnet50

import importlib
import networks
from networks import *

import tensorflow as tf
from fcn.config import cfg

if cfg.TRAIN.SINGLE_FRAME:
    if cfg.NETWORK == 'VGG16':
        __sets['vgg16_convs'] = networks.vgg16_convs(cfg.INPUT, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, cfg.TRAIN.SCALES_BASE, cfg.TRAIN.VERTEX_REG, cfg.TRAIN.TRAINABLE)
    elif cfg.NETWORK == 'VGG16GAN':
        __sets['vgg16_gan'] = networks.vgg16_gan(cfg.INPUT, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, cfg.TRAIN.SCALES_BASE, cfg.TRAIN.VERTEX_REG, cfg.TRAIN.TRAINABLE)
    elif cfg.NETWORK == 'VGG16FLOW':
        __sets['vgg16_flow'] = networks.vgg16_flow(cfg.INPUT, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, cfg.TRAIN.SCALES_BASE, cfg.TRAIN.VERTEX_REG, cfg.TRAIN.TRAINABLE)
    elif cfg.NETWORK == 'VGG16_FLOW_FEATURES':
        __sets['vgg16_flow_features'] = networks.vgg16_flow_features(cfg.INPUT, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, cfg.TRAIN.SCALES_BASE, cfg.TRAIN.VERTEX_REG, cfg.TRAIN.TRAINABLE)
    elif cfg.NETWORK == 'DCGAN':
        __sets['dcgan'] = networks.dcgan()
    elif cfg.NETWORK == 'RESNET50':
        __sets['resnet50'] = networks.resnet50(cfg.INPUT, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.SCALES_BASE)
    elif cfg.NETWORK == 'FCN8VGG':
        __sets['fcn8_vgg'] = networks.fcn8_vgg(cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.MODEL_PATH)
    else:
        net_name = str(cfg.NETWORK)
        __sets[net_name] = locals()[net_name].custom_network()


else:
    if cfg.NETWORK == 'VGG16FLOW':
        __sets['vgg16_flow'] = networks.vgg16_flow(cfg.INPUT, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, cfg.TRAIN.SCALES_BASE, cfg.TRAIN.VERTEX_REG, cfg.TRAIN.TRAINABLE)
    else:
        __sets['vgg16'] = networks.vgg16(cfg.INPUT, cfg.TRAIN.NUM_STEPS, cfg.TRAIN.NUM_CLASSES, cfg.TRAIN.NUM_UNITS, cfg.TRAIN.SCALES_BASE)


def get_network(name):
    """Get a network by name."""
    if not __sets.has_key(str(name).lower()):
        raise KeyError('Unknown network: {}'.format(str(name).lower()))
    return __sets[str(name).lower()]

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
