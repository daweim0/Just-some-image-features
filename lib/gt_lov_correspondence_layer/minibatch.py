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
from normals import gpu_normals

def get_minibatch(roidb, voxelizer):
    """Given a roidb, construct a minibatch sampled from it."""
    # num_images = len(roidb)
    # assert(num_images % cfg.TRAIN.NUM_STEPS == 0), \
    #     'num_images ({}) must be dividable by NUM_STEPS ({})'. \
    #     format(num_images, cfg.TRAIN.NUM_STEPS)

    # Get the input image blob, formatted for tensorflow
    random_scale_ind = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE))
    if cfg.MODE == "test":
        image_left_blob, image_right_blob, flow_blob, occluded_blob, left_label_blob, right_label_blob, depth_blob, warped_blob = _get_image_blob(roidb, random_scale_ind, voxelizer)

        blobs = {'left_image': image_left_blob,
                 'right_image': image_right_blob,
                 'flow': flow_blob,
                 'occluded': occluded_blob,
                 'left_label_blob': left_label_blob,
                 'right_label_blob': right_label_blob,
                 'depth': depth_blob,
                 'warped_im': warped_blob,
                 'roidb': roidb}

    else:
        image_left_blob, image_right_blob, flow_blob, occluded_blob, left_label_blob, right_label_blob = _get_image_blob_training(roidb, random_scale_ind, voxelizer)
        blobs = {'left_image': image_left_blob,
                 'right_image': image_right_blob,
                 'flow': flow_blob,
                 'occluded': occluded_blob,
                 'left_label_blob': left_label_blob,
                 'right_label_blob': right_label_blob}

    # build the label blob
    # depth_blob, label_blob, meta_data_blob, state_blob, weights_blob, points_blob = _get_label_blob(roidb, voxelizer)

    # reshape the blobs
    # num_steps = cfg.TRAIN.NUM_STEPS
    # ims_per_batch = cfg.TRAIN.IMS_PER_BATCH
    #
    # height = im_blob.shape[1]
    # width = im_blob.shape[2]
    # im_blob = im_blob.reshape((num_steps, ims_per_batch, height, width, -1))
    # im_depth_blob = im_depth_blob.reshape((num_steps, ims_per_batch, height, width, -1))
    # im_normal_blob = im_normal_blob.reshape((num_steps, ims_per_batch, height, width, -1))
    #
    # height = label_blob.shape[1]
    # width = label_blob.shape[2]
    # depth_blob = depth_blob.reshape((num_steps, ims_per_batch, height, width, -1))
    # label_blob = label_blob.reshape((num_steps, ims_per_batch, height, width, -1))
    # meta_data_blob = meta_data_blob.reshape((num_steps, ims_per_batch, 1, 1, -1))

    # For debug visualizations
    # _vis_minibatch(im_blob, im_depth_blob, depth_blob, label_blob)

    return blobs

def _get_image_blob(roidb, scale_ind, voxelizer):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb) / 2
    processed_left = []
    processed_right = []
    processed_flow = []
    processed_occluded = []
    processed_depth = []
    processed_left_labels = []
    processed_right_labels = []
    im_scales = []
    processed_warped = []
    for i in xrange(num_images):
        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]

        im_left = pad_im(cv2.imread(roidb[i*2]['image'], cv2.IMREAD_UNCHANGED), 16).astype(np.float32)
        im_left -= cfg.PIXEL_MEANS
        im_left_processed = cv2.resize(im_left, None, None, fx=im_scale, fy=im_scale,
                                       interpolation=cv2.INTER_LINEAR)
        processed_left.append(im_left_processed)

        im_right = pad_im(cv2.imread(roidb[i*2+1]['image'], cv2.IMREAD_UNCHANGED), 16).astype(np.float32)
        im_right -= cfg.PIXEL_MEANS
        im_right_processed = cv2.resize(im_right, None, None, fx=im_scale, fy=im_scale,
                                        interpolation=cv2.INTER_LINEAR)
        processed_right.append(im_right_processed)

        # meta data
        meta_data1 = scipy.io.loadmat(roidb[i*2]['meta_data'])
        K1 = meta_data1['intrinsic_matrix']
        K1 = np.matrix(K1)
        Kinv = np.linalg.inv(K1)
        rt1_inv = se3_inverse(meta_data1['rotation_translation_matrix'])

        meta_data2 = scipy.io.loadmat(roidb[i*2+1]['meta_data'])
        K2 = meta_data2['intrinsic_matrix']
        K2 = np.matrix(K2)
        rt2 = np.matrix(meta_data2['rotation_translation_matrix'])

        # depth
        depth_raw = pad_im(cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED), 16).astype(np.float32)
        processed_depth.append(depth_raw / meta_data1['factor_depth'])
        flow_depth = np.zeros([im_left_processed.shape[0], im_left_processed.shape[1], 2], dtype=np.float32) + [0, 0]
        occluded = np.zeros(flow_depth.shape[:2])
        X = voxelizer.backproject_camera(depth_raw, meta_data1, add_nan=False)
        transform = np.matmul(K2, se3_mul(rt2, rt1_inv))
        Xp = np.matmul(transform, np.append(X, np.ones([1, X.shape[1]], dtype=np.float32), axis=0))
        z = Xp[2] + 1E-15
        x = Xp[0] / z
        y = Xp[1] / z

        end_point_depths = np.zeros([flow_depth.shape[0], flow_depth.shape[1], 3], dtype=np.float32)
        im_left_warped = im_left_processed.copy() * 0

        for a in range(im_left_processed.shape[0]):
            for b in range(im_left_processed.shape[1]):
                index_1d = a*im_left_processed.shape[1]+b
                end_x = int(round(x[0, index_1d]))
                end_y = int(round(y[0, index_1d]))

                if not ((0 <= end_x < im_left_processed.shape[1]) and (0 <= end_y < im_left_processed.shape[0])):
                    flow_depth[a, b] = [x[0, index_1d] - b, y[0, index_1d] - a]
                elif (end_point_depths[end_y, end_x, 0] < z[0, index_1d]) and (end_point_depths[end_y, end_x, 0] != 0):
                    occluded[a, b] = 1
                elif (end_point_depths[end_y, end_x, 0] > z[0, index_1d]) and (end_point_depths[end_y, end_x, 0] != 0):
                    prev_a = int(end_point_depths[end_y, end_x, 1])
                    prev_b = int(end_point_depths[end_y, end_x, 2])
                    occluded[prev_a, prev_b] = 1
                    flow_depth[a, b] = [x[0, index_1d] - b, y[0, index_1d] - a]
                    end_point_depths[end_y, end_x, 0] = z[0, index_1d]
                    im_left_warped[end_y, end_x] = im_left_processed[a, b]
                else:
                    flow_depth[a, b] = [x[0, index_1d] - b, y[0, index_1d] - a]
                    end_point_depths[end_y, end_x, 0] = z[0, index_1d]
                    im_left_warped[end_y, end_x] = im_left_processed[a, b]


        # read left label image
        im = pad_im(cv2.imread(roidb[i*2]['label'], cv2.IMREAD_UNCHANGED), 16)
        # mask the label image according to depth
        if cfg.INPUT == 'DEPTH':
            I = np.where(depth_raw == 0)
            if len(im.shape) == 2:
                im[I[0], I[1]] = 0
            else:
                im[I[0], I[1], :] = 0
        im_cls = _process_label_image(im, roidb[i]['class_colors'], roidb[i]['class_weights'])
        processed_left_labels.append(im_cls)

        # read left label image
        im = pad_im(cv2.imread(roidb[i*2+1]['label'], cv2.IMREAD_UNCHANGED), 16)
        # mask the label image according to depth
        if cfg.INPUT == 'DEPTH':
            I = np.where(depth_raw == 0)
            if len(im.shape) == 2:
                im[I[0], I[1]] = 0
            else:
                im[I[0], I[1], :] = 0
        im_cls = _process_label_image(im, roidb[i]['class_colors'], roidb[i]['class_weights'])
        processed_right_labels.append(im_cls)

        processed_flow.append(flow_depth[:,:,:2])
        processed_occluded.append(occluded)
        processed_warped.append(im_left_warped)

    # Create a blob to hold the input images
    left = im_list_to_blob(processed_left, 3)
    right = im_list_to_blob(processed_right, 3)
    flow = im_list_to_blob(processed_flow, 2)
    occluded = im_list_to_blob(processed_occluded, 1)
    depth = im_list_to_blob(processed_depth, 1)
    left_labels = im_list_to_blob(processed_left_labels, 22)
    right_labels = im_list_to_blob(processed_right_labels, 22)
    warped = im_list_to_blob(processed_warped, 3)

    return left, right, flow, occluded, left_labels, right_labels, depth, warped


def _get_image_blob_training(roidb, scale_ind, voxelizer):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb) / 2
    processed_left = []
    processed_right = []
    processed_flow = []
    processed_occluded = []
    processed_left_labels = []
    processed_right_labels = []
    im_scales = []
    for i in xrange(num_images):
        im_scale = cfg.TRAIN.SCALES_BASE[scale_ind]

        im_left = pad_im(cv2.imread(roidb[i*2]['image'], cv2.IMREAD_UNCHANGED), 16).astype(np.float32)
        im_left -= cfg.PIXEL_MEANS
        im_left_processed = cv2.resize(im_left, None, None, fx=im_scale, fy=im_scale,
                                       interpolation=cv2.INTER_LINEAR)
        processed_left.append(im_left_processed)

        im_right = pad_im(cv2.imread(roidb[i*2+1]['image'], cv2.IMREAD_UNCHANGED), 16).astype(np.float32)
        im_right -= cfg.PIXEL_MEANS
        im_right_processed = cv2.resize(im_right, None, None, fx=im_scale, fy=im_scale,
                                        interpolation=cv2.INTER_LINEAR)
        processed_right.append(im_right_processed)

        # meta data
        meta_data1 = scipy.io.loadmat(roidb[i*2]['meta_data'])
        K1 = meta_data1['intrinsic_matrix']
        K1 = np.matrix(K1)
        Kinv = np.linalg.inv(K1)
        rt1_inv = se3_inverse(meta_data1['rotation_translation_matrix'])

        meta_data2 = scipy.io.loadmat(roidb[i*2+1]['meta_data'])
        K2 = meta_data2['intrinsic_matrix']
        K2 = np.matrix(K2)
        rt2 = np.matrix(meta_data2['rotation_translation_matrix'])

        # depth
        depth_raw = pad_im(cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED), 16).astype(np.float32)
        flow_depth = np.zeros([im_left_processed.shape[0], im_left_processed.shape[1], 2], dtype=np.float32) + [0, 0]
        occluded = np.zeros(flow_depth.shape[:2])
        X = voxelizer.backproject_camera(depth_raw, meta_data1, add_nan=False)
        transform = np.matmul(K2, se3_mul(rt2, rt1_inv))
        Xp = np.matmul(transform, np.append(X, np.ones([1, X.shape[1]], dtype=np.float32), axis=0))
        z = Xp[2] + 1E-15
        x = Xp[0] / z
        y = Xp[1] / z

        end_point_depths = np.zeros([flow_depth.shape[0], flow_depth.shape[1], 3], dtype=np.float32)
        im_left_warped = im_left_processed.copy() * 0

        for a in range(im_left_processed.shape[0]):
            for b in range(im_left_processed.shape[1]):
                index_1d = a*im_left_processed.shape[1]+b
                end_x = int(round(x[0, index_1d]))
                end_y = int(round(y[0, index_1d]))

                if not ((0 <= end_x < im_left_processed.shape[1]) and (0 <= end_y < im_left_processed.shape[0])):
                    flow_depth[a, b] = [x[0, index_1d] - b, y[0, index_1d] - a]
                elif (end_point_depths[end_y, end_x, 0] < z[0, index_1d]) and (end_point_depths[end_y, end_x, 0] != 0):
                    occluded[a, b] = 1
                elif (end_point_depths[end_y, end_x, 0] > z[0, index_1d]) and (end_point_depths[end_y, end_x, 0] != 0):
                    prev_a = int(end_point_depths[end_y, end_x, 1])
                    prev_b = int(end_point_depths[end_y, end_x, 2])
                    occluded[prev_a, prev_b] = 1
                    flow_depth[a, b] = [x[0, index_1d] - b, y[0, index_1d] - a]
                    end_point_depths[end_y, end_x, 0] = z[0, index_1d]
                else:
                    flow_depth[a, b] = [x[0, index_1d] - b, y[0, index_1d] - a]
                    end_point_depths[end_y, end_x, 0] = z[0, index_1d]


        # read left label image
        im = pad_im(cv2.imread(roidb[i*2]['label'], cv2.IMREAD_UNCHANGED), 16)
        # mask the label image according to depth
        if cfg.INPUT == 'DEPTH':
            I = np.where(depth_raw == 0)
            if len(im.shape) == 2:
                im[I[0], I[1]] = 0
            else:
                im[I[0], I[1], :] = 0
        im_cls = _process_label_image(im, roidb[i]['class_colors'], roidb[i]['class_weights'])
        processed_left_labels.append(im_cls)

        # read left label image
        im = pad_im(cv2.imread(roidb[i*2+1]['label'], cv2.IMREAD_UNCHANGED), 16)
        # mask the label image according to depth
        if cfg.INPUT == 'DEPTH':
            I = np.where(depth_raw == 0)
            if len(im.shape) == 2:
                im[I[0], I[1]] = 0
            else:
                im[I[0], I[1], :] = 0
        im_cls = _process_label_image(im, roidb[i]['class_colors'], roidb[i]['class_weights'])
        processed_right_labels.append(im_cls)

        processed_flow.append(flow_depth[:,:,:2])
        processed_occluded.append(occluded)

    # Create a blob to hold the input images
    left = im_list_to_blob(processed_left, 3)
    right = im_list_to_blob(processed_right, 3)
    flow = im_list_to_blob(processed_flow, 2)
    occluded = im_list_to_blob(processed_occluded, 1)
    left_labels = im_list_to_blob(processed_left_labels, 22)
    right_labels = im_list_to_blob(processed_right_labels, 22)

    return left, right, flow, occluded, left_labels, right_labels


# RT is a 3x4 matrix
def se3_inverse(RT):
    R = RT[0:3, 0:3]
    T = RT[0:3, 3].reshape((3,1))
    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = R.transpose()
    RT_new[0:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new

def se3_mul(RT1, RT2):
    R1 = RT1[0:3, 0:3]
    T1 = RT1[0:3, 3].reshape((3,1))

    R2 = RT2[0:3, 0:3]
    T2 = RT2[0:3, 3].reshape((3,1))

    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = np.dot(R1, R2)
    T_new = np.dot(R1, T2) + T1
    RT_new[0:3, 3] = T_new.reshape((3))
    return RT_new


def _process_label_image(label_image, class_colors, class_weights):
    """
    change label image to label index
    """
    height = label_image.shape[0]
    width = label_image.shape[1]
    num_classes = len(class_colors)
    label_index = np.zeros((height, width, num_classes), dtype=np.float32)

    if len(label_image.shape) == 3:
        # label image is in BGR order
        index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
        for i in xrange(len(class_colors)):
            color = class_colors[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            label_index[I[0], I[1], i] = class_weights[i]
    else:
        for i in xrange(len(class_colors)):
            I = np.where(label_image == i)
            label_index[I[0], I[1], i] = class_weights[i]
    
    return label_index


def _get_label_blob(roidb, voxelizer):
    """ build the label blob """

    num_images = len(roidb)
    processed_depth = []
    processed_label = []
    processed_meta_data = []

    for i in xrange(num_images):
        # load meta data
        meta_data = scipy.io.loadmat(roidb[i]['meta_data'])
        im_depth = pad_im(cv2.imread(roidb[i]['depth'], cv2.IMREAD_UNCHANGED), 16)

        # read label image
        im = pad_im(cv2.imread(roidb[i]['label'], cv2.IMREAD_UNCHANGED), 16)
        # mask the label image according to depth
        if cfg.INPUT == 'DEPTH':
            I = np.where(im_depth == 0)
            if len(im.shape) == 2:
                im[I[0], I[1]] = 0
            else:
                im[I[0], I[1], :] = 0
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        im_cls = _process_label_image(im, roidb[i]['class_colors'], roidb[i]['class_weights'])
        processed_label.append(im_cls)

        # depth
        if roidb[i]['flipped']:
            im_depth = im_depth[:, ::-1]
        depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])
        processed_depth.append(depth)

        # voxelization
        if i % cfg.TRAIN.NUM_STEPS == 0:
            points = voxelizer.backproject_camera(im_depth, meta_data)
            voxelizer.voxelized = False
            voxelizer.voxelize(points)
            # store the RT for the first frame
            RT_world = meta_data['rotation_translation_matrix']

        # compute camera poses
        RT_live = meta_data['rotation_translation_matrix']
        pose_world2live = se3_mul(RT_live, se3_inverse(RT_world))
        pose_live2world = se3_inverse(pose_world2live)

        # construct the meta data
        """
        format of the meta_data
        intrinsic matrix: meta_data[0 ~ 8]
        inverse intrinsic matrix: meta_data[9 ~ 17]
        pose_world2live: meta_data[18 ~ 29]
        pose_live2world: meta_data[30 ~ 41]
        voxel step size: meta_data[42, 43, 44]
        voxel min value: meta_data[45, 46, 47]
        """
        K = np.matrix(meta_data['intrinsic_matrix'])
        Kinv = np.linalg.pinv(K)
        mdata = np.zeros(48, dtype=np.float32)
        mdata[0:9] = K.flatten()
        mdata[9:18] = Kinv.flatten()
        mdata[18:30] = pose_world2live.flatten()
        mdata[30:42] = pose_live2world.flatten()
        mdata[42] = voxelizer.step_x
        mdata[43] = voxelizer.step_y
        mdata[44] = voxelizer.step_z
        mdata[45] = voxelizer.min_x
        mdata[46] = voxelizer.min_y
        mdata[47] = voxelizer.min_z
        if cfg.FLIP_X:
            mdata[0] = -1 * mdata[0]
            mdata[9] = -1 * mdata[9]
            mdata[11] = -1 * mdata[11]
        processed_meta_data.append(mdata)

        # compute the delta transformation between frames
        RT_world = RT_live

    # construct the blobs
    height = processed_depth[0].shape[0]
    width = processed_depth[0].shape[1]
    num_classes = voxelizer.num_classes
    depth_blob = np.zeros((num_images, height, width, 1), dtype=np.float32)
    label_blob = np.zeros((num_images, height, width, num_classes), dtype=np.float32)
    meta_data_blob = np.zeros((num_images, 1, 1, 48), dtype=np.float32)
    for i in xrange(num_images):
        depth_blob[i,:,:,0] = processed_depth[i]
        label_blob[i,:,:,:] = processed_label[i]
        meta_data_blob[i,0,0,:] = processed_meta_data[i]

    state_blob = np.zeros((cfg.TRAIN.IMS_PER_BATCH, height, width, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
    weights_blob = np.ones((cfg.TRAIN.IMS_PER_BATCH, height, width, cfg.TRAIN.NUM_UNITS), dtype=np.float32)
    points_blob = np.zeros((cfg.TRAIN.IMS_PER_BATCH, height, width, 3), dtype=np.float32)

    return depth_blob, label_blob, meta_data_blob, state_blob, weights_blob, points_blob


def _vis_minibatch(image_left_blob, image_right_blob, flow_blob, occluded_blob, left_label_blob, right_label_blob, depth_blob, warped_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    for i in range(im_blob.shape[1]):
        for j in xrange(im_blob.shape[0]):
            fig = plt.figure()
            # show image
            im = im_blob[j, i, :, :, :].copy()
            im += cfg.PIXEL_MEANS
            im = im[:, :, (2, 1, 0)]
            im = im.astype(np.uint8)
            fig.add_subplot(221)
            plt.imshow(im)

            # show depth image
            depth = depth_blob[j, i, :, :, 0]
            fig.add_subplot(222)
            plt.imshow(abs(depth))

            # show normal image
            im_normal = im_normal_blob[j, i, :, :, :].copy()
            im_normal += cfg.PIXEL_MEANS
            im_normal = im_normal[:, :, (2, 1, 0)]
            im_normal = im_normal.astype(np.uint8)
            fig.add_subplot(223)
            plt.imshow(im_normal)

            # show label
            label = label_blob[j, i, :, :, :]
            height = label.shape[0]
            width = label.shape[1]
            num_classes = label.shape[2]
            l = np.zeros((height, width), dtype=np.int32)
            for k in xrange(num_classes):
                index = np.where(label[:,:,k] > 0)
                l[index] = k
            fig.add_subplot(224)
            plt.imshow(l)

            plt.show()
