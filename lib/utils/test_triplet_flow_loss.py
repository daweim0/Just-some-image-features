# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Train a FCN"""

from fcn.config import cfg
from gt_data_layer.layer import GtDataLayer
from gt_single_data_layer.layer import GtSingleDataLayer
from gt_flow_data_layer.layer import GtFlowDataLayer
from utils.timer import Timer
import time
import numpy as np
import os
import tensorflow as tf
import sys
import threading
from tensorflow.python import debug as tf_debug

import triplet_flow_loss.triplet_flow_loss_op as triplet_flow_loss_op
from triplet_flow_loss import triplet_flow_loss_op_grad


pause_data_input = False
loader_paused = False


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        # For checkpoint
        self.saver = tf.train.Saver()

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter + 1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

    def train_model(self, sess, train_op, loss, learning_rate, max_iters, net=None, imdb=None, ):
        global pause_data_input
        """Network training loop."""
        # add summary
        tf.summary.tensor_summary('loss', loss)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.output_dir, sess.graph)

        # initialize variables
        print "initializing variables"
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None and str(self.pretrained_model).find('.npy') != -1:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)
        elif self.pretrained_model is not None and str(self.pretrained_model).find('.ckpt') != -1:
            print ('Loading checkpoint from {:s}').format(self.pretrained_model)
            self.saver.restore(sess, self.pretrained_model)

        tf.get_default_graph().finalize()

        last_snapshot_iter = -1
        start_iter = 0
        if self.pretrained_model is not None and str(self.pretrained_model).find('.ckpt') != -1:
            start_index = str(self.pretrained_model).find('iter_') + 5
            end_index = str(self.pretrained_model).find('.ckpt')
            start_iter = int(self.pretrained_model[start_index: end_index])

        loss_history = list()
        timer = Timer()
        for iter in range(start_iter, max_iters):
            timer.tic()
            queue_size = sess.run(net.queue_size_op)

            while sess.run(net.queue_size_op) == 0:
                time.sleep(0.005)

            summary, loss_value, lr, _ = sess.run([merged, loss, learning_rate, train_op])
            train_writer.add_summary(summary, iter)
            timer.toc()

            print 'iter: %d / %d, loss: %7.4f, lr: %.8f, time: %1.2f, queue size before training op: %3i' % \
                  (iter + 1, max_iters, loss_value, lr, timer.diff, queue_size)
            loss_history.append(loss_value)

            if (iter + 1) % cfg.TRAIN.DISPLAY == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time) + ", averaged loss: %7.4f" % np.mean(
                    loss_history)
                loss_history = list()
                # putting any file called show_visuals in the project root directory will show network output
                # in its current (partially trained) state
                if cfg.TRAIN.VISUALIZE_DURING_TRAIN and os.listdir('.').count('show_visuals') != 0:
                    assert net is not None, "the network must be passed to train_model() if VISUALIZE is true"
                    # pause the data loading thread and wait for it to finish
                    pause_data_input = True
                    for i in xrange(1000):
                        time.sleep(0.001)
                        if loader_paused == True:
                            break
                    try:
                        test.test_flow_net(sess, net, imdb, None, n_images=4, training_iter=iter, save_image=False)
                    except IndexError as e:
                        print "error during visualization (training should continue)"
                        print e
                    pause_data_input = False

            if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

                if cfg.TRAIN.DELETE_OLD_CHECKPOINTS:
                    base_dir = os.getcwd()
                    try:
                        os.chdir(self.output_dir)
                        while True:
                            files = sorted(os.listdir("."), key=os.path.getctime)
                            if len(files) < 20:
                                break
                            while files[0].find(".") == -1:
                                files.pop(0)
                            os.remove(files[0])
                    except IndexError:
                        pass
                    finally:
                        os.chdir(base_dir)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    return imdb.roidb


def test_loss(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    dt = 100.0

    _1 = np.zeros([1, 14, 32, 512], dtype=np.float32)
    _2 = np.zeros([1, 14, 32, 512], dtype=np.float32)
    _3 = np.ones([1, 14, 32, 2], dtype=np.float32)
    _4 = np.zeros([1, 14, 32, 1], dtype=np.int32)

    output_1 = run_triplet_flow_op(_1, _2, _3, _4)

    perturbation = np.zeros([1, 14, 32, 512], dtype=np.float32)
    perturbation[0, 13, 31, 0] += dt

    output_2 = run_triplet_flow_op(_1 + perturbation, _2, _3, _4)

    numerical_dL = (output_2[0] - output_1[0]) / dt
    symbolic_dL = output_1[1][0, 0, 0, 0]

    pass


from tensorflow.python.ops import nn_ops

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