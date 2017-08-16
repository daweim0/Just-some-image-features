# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# Heavily modified by David Michelman
# --------------------------------------------------------

"""Train a nerual network"""

from fcn.config import cfg
from gt_flow_data_layer.layer import GtFlowDataLayer
from gt_lov_correspondence_layer.layer import GtLOVFlowDataLayer
from gt_lov_synthetic_layer.layer import GtLOVSyntheticLayer
from utils.timer import Timer
import time
import os
import tensorflow as tf
import threading
from utils.yellowfin import YFOptimizer


pause_data_input = False
loader_paused = False


class SolverWrapper(object):
    """A simple wrapper around Tensorflow's solver. It manages saving checkpoints and deleting old checkpoints.
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
        """Save the network's weights."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

    def train_model(self, sess, train_op, loss, learning_rate, max_iters, net=None):
        """Network training loop."""
        # add summary
        tf.summary.scalar('loss', loss)
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
            start_iter = int(self.pretrained_model[start_index : end_index])

        loss_history = list()
        timer = Timer()
        for iter in range(start_iter, max_iters):
            timer.tic()
            queue_size = sess.run(net.queue_size_op)

            while sess.run(net.queue_size_op) == 0:
                time.sleep(0.005)

            summary, loss_value, lr, _ = sess.run([merged, loss, learning_rate, train_op, ])
            train_writer.add_summary(summary, iter)
            timer.toc()
            
            print 'iter: %d / %d, loss: %7.4f, lr: %0.2e, time: %1.2f, queue size before training op: %3i' %\
                    (iter+1, max_iters, loss_value, lr, timer.diff, queue_size)
            loss_history.append(loss_value)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
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


def load_and_enqueue(sess, net, roidb, num_classes, coord):
    global loader_paused
    assert cfg.TRAIN.OPTICAL_FLOW, "this network can only do optical flow"

    # data layer
    if cfg.IMDB_NAME.count("lov_synthetic") != 0:
        data_layer = GtLOVSyntheticLayer(roidb, num_classes)
    elif cfg.INPUT == "LEFT_RIGHT_CORRESPONDENCE":
        data_layer = GtLOVFlowDataLayer(roidb, num_classes)
    else:
        data_layer = GtFlowDataLayer(roidb, num_classes)

    while not coord.should_stop():
        while pause_data_input:
            loader_paused = True
            time.sleep(0.001)
        loader_paused = False

        while sess.run(net.queue_size_op) > net.queue_size - 1:
            time.sleep(0.01)

        blobs = data_layer.forward()

        left_blob = blobs['left_image']
        right_blob = blobs['right_image']
        flow_blob = blobs['flow']
        occluded_blob = blobs['occluded']
        left_labels = blobs['left_labels']
        right_labels = blobs['right_labels']

        feed_dict = {net.data_left: left_blob, net.data_right: right_blob, net.gt_flow: flow_blob,
                     net.occluded: occluded_blob, net.labels_left: left_labels, net.labels_right: right_labels}

        try:
            sess.run(net.enqueue_op, feed_dict=feed_dict)
        except tf.errors.CancelledError as e:
            print "queue closed, loader thread exiting"
            break

        if sess.run(net.queue_size_op) >18:
            time.sleep(0.0)  # yield to training thread


def train_flow(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000, n_cpu_threads=1):
    """Train a Fast R-CNN network."""
    loss = network.get_output('final_triplet_loss')[0]

    # optimizer
    global_step = tf.Variable(0, trainable=False)
    if cfg.TRAIN.OPTIMIZER.lower() == 'momentumoptimizer' or cfg.TRAIN.OPTIMIZER.lower() == 'momentum':
        starter_learning_rate = cfg.TRAIN.LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
        momentum = cfg.TRAIN.MOMENTUM
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=global_step)

    elif cfg.TRAIN.OPTIMIZER.lower() == 'adam':
        train_op = tf.train.AdamOptimizer(learning_rate=cfg.TRAIN.LEARNING_RATE_ADAM).minimize(loss, global_step=global_step)
        learning_rate = tf.constant(cfg.TRAIN.LEARNING_RATE_ADAM)

    elif cfg.TRAIN.OPTIMIZER.lower() == 'yellowfin':
        # This didn't work at all
        optimizer = YFOptimizer(zero_debias=False, learning_rate=cfg.TRAIN.LEARNING_RATE, momentum=0.0)
        train_op = optimizer.minimize(loss, global_step=global_step)
        learning_rate = optimizer.get_lr_tensor()
    else:
        assert False, "An optimizer must be specified"

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=n_cpu_threads)) as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)

        # thread to load data
        coord = tf.train.Coordinator()
        if cfg.TRAIN.VISUALIZE:
            load_and_enqueue(sess, network, roidb, imdb.num_classes, coord)
        else:
            t = threading.Thread(target=load_and_enqueue, args=(sess, network, roidb, imdb.num_classes, coord))
            t.start()

        print 'Solving...'
        sw.train_model(sess, train_op, loss, learning_rate, max_iters, net = network)
        print 'done solving'

        sess.run(network.close_queue_op)
        coord.request_stop()
        coord.join([t])