# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an imdb (image database)."""

from fcn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
from utils.blob import im_list_to_blob, pad_im, unpad_im
from utils.voxelizer import Voxelizer, set_axes_equal
from utils.se3 import *
import numpy as np
import cv2
import cPickle
import os
import math
import tensorflow as tf
import scipy.io
import time
from normals import gpu_normals
# from pose_estimation import ransac
#from kinect_fusion import kfusion
import cv2
import scipy.ndimage


import triplet_flow_loss.run_slow_flow_calculator_process
# import pyximport; pyximport.install()
import utils.plotting_tools

#output/lov_synthetic_features_drill_crackerbox/batch_size_4_loss_L2_optimizer_ADAM_network_net_labeled_concat_features_2017-08-06_2017-08-06_19-35-28_909447/vgg16_flow_lov_features_iter_9000.ckpt
#output/lov_synthetic_features/batch_size_1_loss_L2_optimizer_ADAM_network_net_labeled_concat_features_2017-08-06_2017-08-06_19:35:48_853842/vgg16_flow_lov_features_iter_69000.ckpt


def get_network_output(sess,net, blob):
    left_blob = blob['left_image']
    depth_blob = blob['depth']
    left_labels_blob = blob['left_labels']

    network_inputs = {net.data_left: left_blob, net.keep_prob: 1.0}

    network_outputs = [net.get_output('features_1x_l'), net.get_output("features_2x_l"),
                       net.get_output("features_4x_l"), net.get_output("features_8x_l")]
    results = siphon_outputs_single_frame(sess, net, network_inputs, network_outputs)

    features_l = np.concatenate([
        scipy.ndimage.zoom(np.squeeze(results[7]), (8, 8, 1), order=1),
        scipy.ndimage.zoom(np.squeeze(results[6]), (4, 4, 1), order=1),
        scipy.ndimage.zoom(np.squeeze(results[5]), (2, 2, 1), order=1),
        np.squeeze(results[0])], axis=2)

    return features_l


def create_voxel(sess, net, imdb, weights_filename, image_set):

    if weights_filename is not None:
        output_dir = get_output_dir(imdb, weights_filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # get a set of all the images that will go into the voxel
    images = {}
    for pair in imdb.roidb:
        if image_set in pair['video_id']:
            if pair['image'] not in images:
                images[pair['image']] = pair

    voxelizer = Voxelizer(1, 1)

    for image in images.values():
        minibatch = get_batch(image, voxelizer)
        features = get_network_output(sess, net, minibatch)
        pass





def display_img(img, title, right=False, ax=None):
    global iiiiii, x_plots, y_plots, axes_left_list, axes_right_list, fig
    if ax is None:
        ax = fig.add_subplot(y_plots, x_plots, iiiiii)
        iiiiii += 1
        if right:
            axes_right_list.append(ax)
        else:
            axes_left_list.append(ax)
    ax.imshow(img)
    ax.set_title(title)
    return ax


def fix_rgb_image(image_in):
    image = image_in.copy() + cfg.PIXEL_MEANS
    image = image[:, :, (2, 1, 0)]
    image = image.astype(np.uint8)
    return image


def siphon_outputs_single_frame(sess, net, data_feed_dict, outputs):
    # compute image blob

    sess.run(net.enqueue_op, feed_dict=data_feed_dict)
    output = sess.run(outputs)

    # assert sess.run(net.queue_size_op) == queue_start_size, "data queue size changed"
    return output


def get_batch(roidb, voxelizer):
    """Given a roidb, construct a minibatch sampled from it."""

    # Get the input image blob, formatted for tensorflow
    image_left_blob, left_label_blob, depth_blob, rt_matrix, intrinsic_matrix = _get_image_blob(roidb)

    blobs = {'left_image': image_left_blob,
             'left_labels': left_label_blob,
             'depth': depth_blob,
             'rt_matrix': rt_matrix,
             'intrinsic_matrix': intrinsic_matrix}

    return blobs


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_left = []
    processed_depth = []
    processed_left_labels = []

    im_left_raw = cv2.imread(roidb['image'], cv2.IMREAD_UNCHANGED)
    im_left = pad_im(im_left_raw[:, :, :3], 16).astype(np.float32)
    im_left -= cfg.PIXEL_MEANS
    im_left_processed = cv2.resize(im_left, None, None, fx=1, fy=1,
                                   interpolation=cv2.INTER_LINEAR)
    processed_left.append(im_left_processed)

    # meta data
    # meta_data1 = scipy.io.loadmat(roidb[i*2]['meta_data'])
    K1 = [[1.06677800e+03, 0.00000000e+00, 3.12986900e+02],
           [0.00000000e+00, 1.06748700e+03, 2.41310900e+02],
           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    K1 = np.matrix(K1)
    rt_mat = np.genfromtxt(roidb['pose'], skip_header=1)

    # depth
    depth_raw = pad_im(cv2.imread(roidb['depth'], cv2.IMREAD_UNCHANGED), 16).astype(np.float32)
    processed_depth.append(depth_raw / 10000)

    if cfg.TRAIN.USE_MASKS:
        # read left label image
        alpha_channel_l = im_left_raw[:, :, 3] / 255.
        im_cls = (np.dstack([np.zeros(im_left_raw.shape[:2]), alpha_channel_l]))
        processed_left_labels.append(im_cls)
    else:
        processed_left_labels.append(np.zeros([im_left_processed.shape[0], im_left_processed.shape[1],2], dtype=np.int32) + [0, 1])


    # Create a blob to hold the input images
    left = im_list_to_blob(processed_left, 3)
    depth = im_list_to_blob(processed_depth, 1)
    if cfg.TRAIN.USE_MASKS:
        left_labels = im_list_to_blob(processed_left_labels, processed_left_labels[0].shape[2])
    else:
        left_labels = im_list_to_blob(processed_left_labels, 2, mess_up_single_channel=True)

    return left, left_labels, depth, K1, rt_mat


# RT is a 3x4 matrix
def se3_inverse(RT):
    R = RT[0:3, 0:3]
    T = RT[0:3, 3].reshape((3, 1))
    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = R.transpose()
    RT_new[0:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new


def se3_mul(RT1, RT2):
    R1 = RT1[0:3, 0:3]
    T1 = RT1[0:3, 3].reshape((3, 1))

    R2 = RT2[0:3, 0:3]
    T2 = RT2[0:3, 3].reshape((3, 1))

    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = np.dot(R1, R2)
    T_new = np.dot(R1, T2) + T1
    RT_new[0:3, 3] = T_new.reshape((3))
    return RT_new

