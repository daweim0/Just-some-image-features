#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# Heavily modified by David Michelman
# --------------------------------------------------------

"""Test a FCN on an image database."""

import _init_paths
# from fcn.test import test_net, test_gan
# from fcn.test import test_net_single_frame
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
    parser = argparse.ArgumentParser(description='Test a network\'s ability to generate dense matchable features')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use', default=0, type=int)
    parser.add_argument('--weights', dest='pretrained_model', help='pretrained model', default=None, type=str)
    parser.add_argument('--model', dest='model', help='model to test', default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file', help='the config file used when training', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name', help='dataset to test', default=None, type=str)
    parser.add_argument('--network', dest='network_name', help='name of the network', default=None, type=str)
    parser.add_argument('--n_cpu_threads', dest='n_cpu_threads', help='passed to tensorflow', default=10, type=int)
    parser.add_argument('--show_correspondence', dest='show_correspondence', help='draw lines between corresponding points', default='False', type=str)
    parser.add_argument('--calc_EPE_all', dest='calc_EPE_all', help='calculate EPE for all testing data', default='False', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cfg.MODE = "test"
    args = parse_args()
    print('Called with args:')
    print(args)

    # read the model cfg file so that network settings are properly restored
    # pixel means is a numpy array (which is hard to parse and not used). Remove it then parse the cfg file
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

    weights_filename = os.path.splitext(os.path.basename(args.model))[0]

    cfg.IMDB_NAME = args.imdb_name
    imdb = get_imdb(args.imdb_name)

    cfg.GPU_ID = args.gpu_id
    device_name = '/gpu:{:d}'.format(args.gpu_id)
    print "using device " + device_name

    cfg.TRAIN.NUM_STEPS = 1
    cfg.TRAIN.GRID_SIZE = cfg.TEST.GRID_SIZE
    if cfg.NETWORK == 'FCN8VGG':
        path = osp.abspath(osp.join(cfg.ROOT_DIR, args.pretrained_model))
        cfg.TRAIN.MODEL_PATH = path
    cfg.TRAIN.TRAINABLE = False

    from networks.factory import get_network
    network = get_network(cfg.NETWORK)
    print '\n# Using network `{:s}` for testing (from folder `{:s}`'.format(args.network_name, args.model)

    # start a session
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=args.n_cpu_threads))
    saver.restore(sess, args.model)

    test_flow_net(sess, network, imdb, weights_filename, show_arrows=args.show_correspondence=='True', calculate_EPE_all_data=args.calc_EPE_all=='True')

