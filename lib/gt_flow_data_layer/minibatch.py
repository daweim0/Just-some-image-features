# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import sys
import numpy as np
import numpy.random as npr
import cv2
from fcn.config import cfg
from utils.blob import im_list_to_blob, pad_im, chromatic_transform
from utils.se3 import *
import scipy.io
from utils import sintel_utils


preloaded_images = {}

OBSERVE_OCCLUSIONS = True
iter_num = 0

def preload_data(roidb):
    random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
    _get_image_blob(roidb, random_scale_ind)


def get_minibatch(roidb, voxelizer):
    """Given a roidb, construct a minibatch sampled from it."""

    # Get the input image blob, formatted for tensorflow
    random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
    image_left_blob, image_right_blob, flow_blob, occluded_blob, image_scales = _get_image_blob(roidb, random_scale_ind)

    # build the label blob
    # depth_blob, label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, \
    #     gan_z_blob = _get_label_blob(roidb, voxelizer)

    # For debug visualizations
    if cfg.TRAIN.VISUALIZE:
        _vis_minibatch(image_left_blob, image_right_blob, flow_blob)

    blobs = {'left_image': image_left_blob,
             'right_image': image_right_blob,
             'flow': flow_blob,
             'occluded': occluded_blob}

    return blobs


def _get_image_blob(roidb, scale_ind):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    global iter_num
    iter_num += 1

    num_images = len(roidb)
    processed_left = []
    processed_right = []
    processed_flow = []
    processed_occluded = []
    im_scales = []
    # if cfg.TRAIN.GAN:
    #     processed_ims_rescale = []

    for i in xrange(num_images):
        # # meta data
        # meta_data = scipy.io.loadmat(roidb[i]['meta_data'])
        # K = meta_data['intrinsic_matrix'].astype(np.float32, copy=True)
        # fx = K[0, 0]
        # fy = K[1, 1]
        # cx = K[0, 2]
        # cy = K[1, 2]

        # # depth raw
        # im_depth_raw = pad_im(cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED), 16)
        # height = im_depth_raw.shape[0]
        # width = im_depth_raw.shape[1]

        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]
        im_scales.append(im_scale)

        if not cfg.PUPPER_DATASET:
            # left image
            # if roidb[i]['image_left'] not in preloaded_images:
            im_left = pad_im(cv2.imread(roidb[i]['image_left'], cv2.IMREAD_UNCHANGED), 16)
            if im_left.shape[2] == 4:
                im = np.copy(im_left[:, :, :3])
                alpha = im_left[:, :, 3]
                I = np.where(alpha == 0)
                im[I[0], I[1], :] = 0
                im_left = im
            im_left_orig = im_left.astype(np.float32, copy=True)
            if cfg.NORMALIZE_IMAGES:
                im_left_processed = (im_left_orig - im_left_orig.mean()) / im_left_orig.std()
            else:
                im_left_orig -= cfg.PIXEL_MEANS
                im_left_processed = cv2.resize(im_left_orig, None, None, fx=im_scale, fy=im_scale,
                                                interpolation=cv2.INTER_LINEAR)

            #     preloaded_images[roidb[i]['image_left']] = im_left_processed
            # im_left_processed = preloaded_images[roidb[i]['image_left']]

            # if roidb[i]['image_right'] not in preloaded_images:
            im_right = pad_im(cv2.imread(roidb[i]['image_right'], cv2.IMREAD_UNCHANGED), 16)
            if im_right.shape[2] == 4:
                im = np.copy(im_right[:, :, :3])
                alpha = im_right[:, :, 3]
                I = np.where(alpha == 0)
                im[I[0], I[1], :] = 0
                im_right = im
            im_right_orig = im_right.astype(np.float32, copy=True)
            if cfg.NORMALIZE_IMAGES:
                im_right_processed = (im_right_orig - im_right_orig.mean()) / im_right_orig.std()
            else:
                im_right_orig -= cfg.PIXEL_MEANS
                im_right_processed = cv2.resize(im_right_orig, None, None, fx=im_scale, fy=im_scale,
                                                interpolation=cv2.INTER_LINEAR)
            #     preloaded_images[roidb[i]['image_right']] = im_right_processed
            # im_right_processed = preloaded_images[roidb[i]['image_right']]

            # if roidb[i]['flow'] not in preloaded_images:
            gt_flow = pad_im(sintel_utils.read_flow_file_with_path(roidb[i]['flow']).transpose([1, 0, 2]), 16)
            flow_processed = cv2.resize(gt_flow, None, None, fx=im_scale/cfg.NET_CONF.MATCHING_STAGE_SCALE,
                                        fy=im_scale/cfg.NET_CONF.MATCHING_STAGE_SCALE, interpolation=cv2.INTER_LINEAR)
            flow_processed *= cfg.TRAIN.SCALES_BASE[scale_ind] / cfg.NET_CONF.MATCHING_STAGE_SCALE

            # mask = occlusions_processed.sum(axis=2)
            # mask = np.dstack((mask, mask))
            # flow_processed = np.where(mask, flow_processed, np.nan)
            # #TODO: remove this debugging check
            # np.array_equal((mask == False).nonzero(), (np.isnan(flow_processed)).nonzero())

            #     preloaded_images[roidb[i]['flow']] = flow_processed
            # flow_processed = preloaded_images[roidb[i]['flow']]

            # if roidb[i]['occluded'] not in preloaded_images:
            occlusions = pad_im(cv2.imread(roidb[i]['occluded']), 16)
            occlusions_processed = cv2.resize(occlusions, None, None, fx=im_scale/cfg.NET_CONF.MATCHING_STAGE_SCALE,
                                              fy=im_scale/cfg.NET_CONF.MATCHING_STAGE_SCALE, interpolation=cv2.INTER_LINEAR).sum(axis=2) / (255 * 3)
            occluded_processed = np.round(occlusions_processed).astype(np.int32)
                # preloaded_images[roidb[i]['occluded']] = occluded_processed
            # occluded_processed = preloaded_images[roidb[i]['occluded']]

            processed_left.append(im_left_processed)
            processed_right.append(im_right_processed)
            # processed_flow.append(pad_im(flow_processed, 16))
            # processed_occluded.append(pad_im(occluded_processed, 16))
            processed_flow.append(flow_processed)
            processed_occluded.append(occluded_processed)

        else:
            if "pupper" not in preloaded_images:
                im_left = pad_im(cv2.imread("data/pupper_dataset/pupper.png", cv2.IMREAD_UNCHANGED)[...,:3], 16)
                im_left_orig = im_left.astype(np.float32, copy=True)
                im_left_orig -= cfg.PIXEL_MEANS
                im_left_processed = cv2.resize(im_left_orig, None, None, fx=im_scale, fy=im_scale,
                                               interpolation=cv2.INTER_LINEAR)
            #     preloaded_images["pupper"] = im_left_processed
            # pup = preloaded_images["pupper"]
            pup = im_left_processed

            try:
                image_height = int(400/16)*16
                image_width = int(600/16)*16
                flow_size = 6
                noise = hash(str(float(iter_num) + 3.14159001))
                x_flow = noise % (2*flow_size+1) - flow_size
                y_flow = (noise / (2*flow_size+1)) % (2*flow_size+1) - flow_size
                # x_flow = 0
                # y_flow = 0
                x_start = noise % (pup.shape[1] - image_width - 22) + 11
                y_start = (noise / 50) % (pup.shape[0] - image_height - 22) + 11
                pup_left = pup[y_start + y_flow:y_start + y_flow + image_height,
                           x_start + x_flow:x_start + x_flow + image_width, :]
                pup_right = pup[y_start:y_start + image_height,
                           x_start:x_start + image_width, :]
                flow = np.dstack([np.zeros([image_height, image_width], dtype=np.float32) + x_flow,
                                  np.zeros([image_height, image_width], dtype=np.float32) + y_flow])
            except:
                print "\N\N\N\N\N\TERROR CREATING PUPPER FLOW\N\N\N\N"
                pup_left = pup[:image_height, :image_width, :]
                pup_right = pup[:image_height, :image_width, :]
                flow = np.zeros([image_height, image_width, 2], dtype=np.float32)

            occluded = np.zeros([image_height, image_width], dtype=np.int32)

            processed_left.append(pup_left)
            processed_right.append(pup_right)
            # processed_flow.append(pad_im(flow_processed, 16))
            # processed_occluded.append(pad_im(occluded_processed, 16))
            processed_flow.append(flow)
            processed_occluded.append(occluded)

    # Create a blob to hold the input images
    image_left_blob = im_list_to_blob(processed_left, 3)
    image_right_blob = im_list_to_blob(processed_right, 3)
    gt_flow_blob = im_list_to_blob(processed_flow, 2)
    occluded_blob = im_list_to_blob(processed_occluded, 1)
    # if cfg.TRAIN.GAN:
    #     blob_rescale = im_list_to_blob(processed_ims_rescale, 3)
    # else:
    blob_rescale = []

    return image_left_blob, image_right_blob, gt_flow_blob, occluded_blob, im_scales


def _vis_minibatch(image_left_blob, image_right_blob, flow_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    for i in xrange(image_left_blob.shape[0]):
        fig = plt.figure()
        # show left
        im_left = image_left_blob[i, :, :, :].copy()
        if cfg.NORMALIZE_IMAGES:
            im_left = (im_left - im_left.min()) / (im_left.max() - im_left.min())
        else:
            im_left += cfg.PIXEL_MEANS
            im_left = im_left[:, :, (2, 1, 0)]
            im_left = im_left.astype(np.uint8)
        fig.add_subplot(221)
        plt.imshow(im_left)

        # show right
        im_right = image_right_blob[i, :, :, :].copy()
        if cfg.NORMALIZE_IMAGES:
            im_right = (im_right - im_right.min()) / (im_right.max() - im_right.min())
        else:
            im_right += cfg.PIXEL_MEANS
            im_right = im_right[:, :, (2, 1, 0)]
            im_right = im_right.astype(np.uint8)
        fig.add_subplot(222)
        plt.imshow(im_right)

        # show normal image
        im_flow = flow_blob[i, :, :].copy()
        fig.add_subplot(223)
        plt.imshow(sintel_utils.sintel_compute_color(im_flow))

        plt.show()