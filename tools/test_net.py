#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an image database."""

import _init_paths
from fcn.test import test_net, test_gan
from fcn.test import test_net_single_frame
from fcn.test import test_flow_net
from fcn.config import cfg, cfg_from_file, cfg_from_string
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import tensorflow as tf
import os.path as osp
import ast

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use', default=0, type=int)
    parser.add_argument('--weights', dest='pretrained_model', help='pretrained model', default=None, type=str)
    parser.add_argument('--model', dest='model', help='model to test', default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait', help='wait until net file exists', default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name', help='dataset to test', default='shapenet_scene_val', type=str)
    parser.add_argument('--network', dest='network_name', help='name of the network', default=None, type=str)
    parser.add_argument('--rig', dest='rig_name', help='name of the camera rig file', default=None, type=str)
    parser.add_argument('--kfusion', dest='kfusion', help='run kinect fusion or not', default=False, type=bool)
    parser.add_argument('--n_cpu_threads', dest='n_cpu_threads', help='self explanatory', default=2, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    # pixel means is a numpy array (which ast can't parse)
    malformed_model_cfg = (open(args.model.replace(args.model.split("/")[-1], "") + "config.txt").read()
                           .replace("\n", "").replace("\t", "").replace("          ", " "))
    pixel_means_index = malformed_model_cfg.find('PIXEL_MEANS') - 1
    after_str = malformed_model_cfg[pixel_means_index:-1]
    end_index = after_str.find("),")
    cfg_safe = malformed_model_cfg[0:pixel_means_index] + after_str[end_index+2:] + "}"

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg_from_string(cfg_safe)

    print('Using config:')
    pprint.pprint(cfg)
    # print('The arguments were updated from the passed model:', weights_processed.keys())

    weights_filename = os.path.splitext(os.path.basename(args.model))[0]

    imdb = get_imdb(args.imdb_name)

    cfg.GPU_ID = args.gpu_id
    device_name = '/gpu:{:d}'.format(args.gpu_id)
    print device_name

    cfg.TRAIN.NUM_STEPS = 1
    cfg.TRAIN.GRID_SIZE = cfg.TEST.GRID_SIZE
    if cfg.NETWORK == 'FCN8VGG':
        path = osp.abspath(osp.join(cfg.ROOT_DIR, args.pretrained_model))
        cfg.TRAIN.MODEL_PATH = path
    cfg.TRAIN.TRAINABLE = False

    from networks.factory import get_network
    network = get_network(args.network_name)
    print '\n# Using network `{:s}` for testing (from folder `{:s}`'.format(args.network_name, args.model)

    # start a session
    saver = tf.train.Saver()
    if args.kfusion:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    else:
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=args.n_cpu_threads))
    saver.restore(sess, args.model)
    print ('# Loading model weights from {:s}').format(args.model)

    if cfg.TEST.OPTICAL_FLOW:
        test_flow_net(sess, network, imdb, weights_filename)
    elif cfg.TEST.SINGLE_FRAME:
        if cfg.TEST.GAN:
            test_gan(sess, network, imdb, weights_filename)
        else:
            test_net_single_frame(sess, network, imdb, weights_filename, args.rig_name, args.kfusion)
    else:
        test_net(sess, network, imdb, weights_filename, args.rig_name, args.kfusion)
