# output/sintel_albedo_features/batch_size_1_loss_L2_optimizer_MomentumOptimizer_conv_size_1_ConcatSub_append_n_convolutions_1_2017-06-27/vgg16_flow_sintel_albedo_iter_15000.ckpt
# data/imagenet_models/vgg16_convs.npy
# output/sintel_albedo_features_trainable_false/batch_size_1_loss_L2_optimizer_MomentumOptimizer_conv_size_1_ConcatSub_append_n_convolutions_1_2017-06-28/vgg16_flow_sintel_albedo_iter_1000.ckpt

#output/pupper_features_trainable_false/batch_size_1_loss_L2_optimizer_MomentumOptimizer_conv_size_1_ConcatSub_append_n_convolutions_1_2017-06-28/vgg16_flow_sintel_albedo_iter_20000.ckpt


import pyximport
# pyximport.install()
from fcn.config import cfg
# from gt_data_layer.layer import GtDataLayer
# from gt_single_data_layer.layer import GtSingleDataLayer
from gt_flow_data_layer.layer import GtFlowDataLayer
from utils.timer import Timer
import time
import numpy as np
import os
import tensorflow as tf
import sys
import threading
# from tensorflow.python import debug as tf_debug
# from triplet_flow_loss import slow_flow_calculator
# from triplet_flow_loss.slow_flow_calculator_cython import compute_flow
import sintel_utils
from triplet_flow_loss.run_slow_flow_calculator_process import get_flow_parallel

# import triplet_flow_loss.triplet_flow_loss_op as triplet_flow_loss_op
# from triplet_flow_loss import triplet_flow_loss_op_grad

import matplotlib.pyplot as plt

i = 1
plot_x = 3
plot_y = 3


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    return imdb.roidb


def test_loss(network, roidb, pretrained_model=None):
    """Train a Fast R-CNN network."""
    # dt = 100.0
    #
    # _1 = np.zeros([1, 14, 32, 512], dtype=np.float32)
    # _2 = np.zeros([1, 14, 32, 512], dtype=np.float32)
    # _3 = np.ones([1, 14, 32, 2], dtype=np.float32)
    # _4 = np.zeros([1, 14, 32, 1], dtype=np.int32)
    #
    # output_1 = run_triplet_flow_op(_1, _2, _3, _4)
    #
    # perturbation = np.zeros([1, 14, 32, 512], dtype=np.float32)
    # perturbation[0, 13, 31, 0] += dt
    #
    # output_2 = run_triplet_flow_op(_1 + perturbation, _2, _3, _4)
    #
    # numerical_dL = (output_2[0] - output_1[0]) / dt
    # symbolic_dL = output_1[1][0, 0, 0, 0]

    global i
    # plt.ion()
    cfg.TRAIN.IMS_PER_BATCH = 1

    data_layer = GtFlowDataLayer(roidb, None)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # initialize variables
        print "initializing variables"
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        if pretrained_model is not None and str(pretrained_model).find('.npy') != -1:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            network.load(pretrained_model, sess, True)
        elif pretrained_model is not None and str(pretrained_model).find('.ckpt') != -1:
            print ('Loading checkpoint from {:s}').format(pretrained_model)
            saver.restore(sess, pretrained_model)
        tf.get_default_graph().finalize()

        upscore_matcher_input = [network.get_output('upscore_l'), network.get_output('upscore_r'),
                                 network.get_output('gt_flow'), network.get_output('occluded')]

    #     score_conv_4_matcher_input = [network.get_output('score_conv4_l'), network.get_output('score_conv4_r'),
    #                                   network.get_output('flow_upscore_4'), network.get_output('occluded_pool3')]
    #
    #     conv_5_matcher_input = [network.get_output('conv5_3_l'), network.get_output('conv5_3_r'),
    #                             network.get_output('flow_pool3_out'), network.get_output('occluded_pool4')]
    #
    #     upscore_conv_5_matcher_input = [network.get_output('upscore_conv5_l'), network.get_output('upscore_conv5_r'),
    #                                     network.get_output('flow_upscore_4'), network.get_output('occluded_pool3')]
    #
    #     conv_3_matcher_input = [network.get_output('conv3_l'), network.get_output('conv3_r'),
    #                             network.get_output('flow_pool2'), network.get_output('occluded_pool2')]
    #
    #     conv_1_matcher_input = [network.get_output('conv1_l'), network.get_output('conv1_r'),
    #                             network.get_output('gt_flow'), network.get_output('occluded')]

        while True:
            # feed data and get feature map
            blobs = data_layer.forward()
            left_blob = blobs['left_image']
            right_blob = blobs['right_image']
            flow_blob = blobs['flow']
            occluded_blob = blobs['occluded']
            feed_dict = {network.data_left: left_blob, network.data_right: right_blob, network.gt_flow: flow_blob,
                         network.occluded: occluded_blob, network.keep_prob: 1.0}
            sess.run(network.enqueue_op, feed_dict=feed_dict)

            # left_features, right_features, flow, mask = sess.run([network.get_output('upscore_l'),
            #           network.get_output('upscore_r'), network.get_output('gt_flow'), network.get_output('occluded')])

            left_features, right_features, gt_flow, mask = sess.run(upscore_matcher_input)

            # left_features, right_features, gt_flow, mask = left_blob[0], right_blob[0], flow_blob[0], occluded_blob[0]

            left_features = np.squeeze(left_features)
            right_features = np.squeeze(right_features)
            gt_flow = np.squeeze(gt_flow)
            mask = np.squeeze(mask)
            print "starting"
            predicted_flow = get_flow_parallel(left_features, right_features, mask, neighborhood_len_import=200)
            # predicted_flow = compute_flow(left_features, right_features, mask, neighborhood_len_import=40)
            # predicted_flow = slow_flow_calculator.compute_flow(left_features, right_features, mask)

            EPE = sintel_utils.calculate_EPE(gt_flow, predicted_flow)
            print "average EPE: " + str(EPE)

            scale_low, scale_high = sintel_utils.colorize_features(left_features, get_scale=True)

            plot_stuff(fix_rgb_image(left_blob[0]), "left image")
            plot_stuff(fix_rgb_image(right_blob[0]), "right image")
            plot_stuff(sintel_utils.raw_color_from_flow(gt_flow), "gt flow")
            plot_stuff(sintel_utils.colorize_features(left_features, scale_low=scale_low, scale_high=scale_high), "left features")
            plot_stuff(sintel_utils.colorize_features(right_features), "right features")
            plot_stuff(sintel_utils.raw_color_from_flow(predicted_flow), "predicted flow")
            # diff_zero = left_features[10:-10, 10:-10] - right_features[10:-10, 10:-10]
            flow_means = [gt_flow[:,:,0].mean(), gt_flow[:,:,1].mean()]
            diff_flowd = left_features[10-int(flow_means[1]):-10-int(flow_means[1]), 10-int(flow_means[0]):-10-int(flow_means[0])] - right_features[10:-10, 10:-10]
            # plot_stuff(colorize_features(np.abs(diff_zero)), "feature differences no shift")
            plot_stuff(sintel_utils.colorize_features(np.abs(diff_flowd), scale_high=scale_high), "feature differences shifted by flow")

            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.09, hspace=0.05)
            plt.show()

            # input = raw_input("press enter to see the next plot (or type exit to exit)")
            # if input == "exit":
            #     break
            clear_plot()
            i = 1


def fix_rgb_image(image_in):
    image = image_in.copy() + cfg.PIXEL_MEANS
    image = image[:, :, (2, 1, 0)]
    image = image.astype(np.uint8)
    return image


def plot_stuff(array, title=""):
    global i
    ax = plt.subplot(plot_y, plot_x, i)
    ax.imshow(array, interpolation='nearest')
    ax.set_title(title)
    i += 1


def clear_plot():
    global i
    i = 1
    plt.clf()


def run_triplet_flow_op(left, right, flow, occluded, margin=1):
    left_tensor = tf.constant(left)
    right_tensor = tf.constant(right)
    flow_tensor = tf.constant(flow)
    occluded_tensor = tf.constant(occluded)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        op = triplet_flow_loss_op.triplet_flow_loss(left_tensor, right_tensor, flow_tensor, occluded_tensor, margin=margin)
        return sess.run(op)