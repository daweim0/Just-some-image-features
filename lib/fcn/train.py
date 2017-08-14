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
from gt_lov_correspondence_layer.layer import GtLOVFlowDataLayer
from gt_lov_synthetic_layer.layer import GtLOVSyntheticLayer
from utils.timer import Timer
import time
import numpy as np
import os
import tensorflow as tf
import sys
import threading
from tensorflow.python import debug as tf_debug
import test
from utils.yellowfin import YFOptimizer


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
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)


    def train_model(self, sess, train_op, loss, learning_rate, max_iters, net=None, imdb = None, momentum_op=None):
        global pause_data_input
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

            if momentum_op != None:
                summary, loss_value, lr, _, momentum = sess.run([merged, loss, learning_rate, train_op, momentum_op,])
            else:
                summary, loss_value, lr, _ = sess.run([merged, loss, learning_rate, train_op, ])
                momentum = cfg.TRAIN.MOMENTUM
            train_writer.add_summary(summary, iter)
            timer.toc()
            
            print 'iter: %d / %d, loss: %7.4f, lr: %0.2e, momentum: %2.2f, time: %1.2f, queue size before training op: %3i' %\
                    (iter+1, max_iters, loss_value, lr, momentum, timer.diff, queue_size)
            loss_history.append(loss_value)

            if (iter+1) % cfg.TRAIN.DISPLAY == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time) + ", averaged loss: %7.4f" % np.mean(loss_history)
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


    def train_gan(self, sess, train_op_d, train_op_g, loss_d, loss_true, loss_false, loss_g, loss_l1, loss_ad, max_iters):
        """Network training loop."""

        # initialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        tf.get_default_graph().finalize()

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            timer.tic()
            # update discriminator
            loss_value_d, loss_value_true, loss_value_false, _ \
                = sess.run([loss_d, loss_true, loss_false, train_op_d])

            # update generator
            loss_value_g, loss_value_l1, loss_value_ad, _ = sess.run([loss_g, loss_l1, loss_ad, train_op_g])
            timer.toc()
            
            #print 'iter: %d / %d, loss_d: %.4f, loss_true: %.4f, loss_false: %.4f, lr_d: %.8f, time: %.2f' %\
            #        (iter+1, max_iters, loss_value_d, loss_value_true, loss_value_false, lr_d, timer.diff)
            print 'iter: %d / %d, loss_d: %.4f, loss_true: %.4f, loss_false: %.4f, loss_g: %.4f, loss_l1: %.4f, loss_ad: %.4f, time: %.2f' %\
                    (iter+1, max_iters, loss_value_d, loss_value_true, loss_value_false, loss_value_g, loss_value_l1, loss_value_ad, timer.diff)

            if (iter+1) % (10 * cfg.TRAIN.DISPLAY) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


    def train_model_vertex(self, sess, train_op, loss, loss_cls, loss_vertex, learning_rate, max_iters):
        """Network training loop."""
        # add summary
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.output_dir, sess.graph)

        # intialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        tf.get_default_graph().finalize()

        # tf.train.write_graph(sess.graph_def, self.output_dir, 'model.pbtxt')

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            timer.tic()
            summary, loss_value, loss_cls_value, loss_vertex_value, lr, _ = sess.run([merged, loss, loss_cls, loss_vertex, learning_rate, train_op])
            train_writer.add_summary(summary, iter)
            timer.toc()
            
            print 'iter: %d / %d, loss: %.4f, loss_cls: %.4f, loss_vertex: %.4f, lr: %.8f, time: %.2f' %\
                    (iter+1, max_iters, loss_value, loss_cls_value, loss_vertex_value, lr, timer.diff)

            if (iter+1) % (10 * cfg.TRAIN.DISPLAY) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

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

    i = 0
    while not coord.should_stop():
        while pause_data_input:
            loader_paused = True
            time.sleep(0.001)
        loader_paused = False

        while sess.run(net.queue_size_op) > net.queue_size - 1:
            time.sleep(0.01)

        # print "\t\t\t\t\t\tstarting load operation " + str(i)

        blobs = data_layer.forward()

        left_blob = blobs['left_image']
        right_blob = blobs['right_image']
        flow_blob = blobs['flow']
        occluded_blob = blobs['occluded']
        left_labels = blobs['left_labels']
        right_labels = blobs['right_labels']

        feed_dict = {net.data_left: left_blob, net.data_right: right_blob, net.gt_flow: flow_blob,
                     net.occluded: occluded_blob, net.labels_left: left_labels, net.labels_right: right_labels,
                     net.keep_prob: 0.5}

        # print "\t\t\t\t\t\trunning enqueue op " + str(i)
        try:
            sess.run(net.enqueue_op, feed_dict=feed_dict)
            # print "\t\t\t\t\t\tfinished enqueue op " + str(i) + " queue size is now " + str(sess.run(net.queue_size_op))
        except tf.errors.CancelledError as e:
            print "queue closed, loader thread exiting"
            break
        i += 1
        if sess.run(net.queue_size_op) >18:
            time.sleep(0.0)  # yeild to training thread

def loss_cross_entropy(scores, labels):
    """
    scores: a list of tensors [batch_size, height, width, num_classes]
    labels: a list of tensors [batch_size, height, width, num_classes]
    """

    with tf.name_scope('loss'):
        loss = 0
        for i in range(cfg.TRAIN.NUM_STEPS):
            score = scores[i]
            label = labels[i]
            cross_entropy = -tf.reduce_sum(label * score, reduction_indices=[3])
            loss += tf.div(tf.reduce_sum(cross_entropy), tf.reduce_sum(label))
        loss /= cfg.TRAIN.NUM_STEPS
    return loss

def loss_cross_entropy_single_frame(scores, labels):
    """
    scores: a tensor [batch_size, height, width, num_classes]
    labels: a tensor [batch_size, height, width, num_classes]
    """

    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(labels * scores, reduction_indices=[3])
        loss = tf.div(tf.reduce_sum(cross_entropy), tf.reduce_sum(labels))

    return loss


def train_flow(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000, n_cpu_threads=1):
    """Train a Fast R-CNN network."""
    # loss = network.get_output('final_triplet_loss').loss
    loss = network.get_output('final_triplet_loss')[0]

    # optimizer
    global_step = tf.Variable(0, trainable=False)
    if cfg.TRAIN.OPTIMIZER.lower() == 'momentumoptimizer' or cfg.TRAIN.OPTIMIZER.lower() == 'momentum':
        starter_learning_rate = cfg.TRAIN.LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
        momentum = cfg.TRAIN.MOMENTUM
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=global_step)
        momentum_op = None

    elif cfg.TRAIN.OPTIMIZER.lower() == 'adam':
        train_op = tf.train.AdamOptimizer(learning_rate=cfg.TRAIN.LEARNING_RATE_ADAM).minimize(loss, global_step=global_step)
        learning_rate = tf.constant(cfg.TRAIN.LEARNING_RATE_ADAM)
        momentum_op = None

    elif cfg.TRAIN.OPTIMIZER.lower() == 'yellowfin':
        optimizer = YFOptimizer(zero_debias=False, learning_rate=cfg.TRAIN.LEARNING_RATE, momentum=0.0)
        train_op = optimizer.minimize(loss, global_step=global_step)
        learning_rate = optimizer.get_lr_tensor()
        momentum_op = optimizer.get_mu_tensor()
    else:
        assert False, "An optimizer must be specified"

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=n_cpu_threads)) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)

        # thread to load data
        coord = tf.train.Coordinator()
        if cfg.TRAIN.VISUALIZE:
            load_and_enqueue(sess, network, roidb, imdb.num_classes, coord)
        else:
            t = threading.Thread(target=load_and_enqueue, args=(sess, network, roidb, imdb.num_classes, coord))
            t.start()

        print 'Solving...'
        sw.train_model(sess, train_op, loss, learning_rate, max_iters, net = network, imdb=imdb, momentum_op=momentum_op)
        print 'done solving'

        sess.run(network.close_queue_op)
        coord.request_stop()
        coord.join([t])


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    if cfg.TRAIN.SINGLE_FRAME:
        # classification loss
        if cfg.NETWORK == 'FCN8VGG':
            scores = network.prob
            labels = network.gt_label_2d_queue
            loss = loss_cross_entropy_single_frame(scores, labels)
        else:
            if cfg.TRAIN.VERTEX_REG:
                scores = network.get_output('prob')
                labels = network.get_output('gt_label_2d')
                loss_cls = loss_cross_entropy_single_frame(scores, labels)

                vertex_pred = network.get_output('vertex_pred')
                vertex_targets = network.get_output('vertex_targets')
                vertex_weights = network.get_output('vertex_weights')
                loss_vertex = tf.div( tf.reduce_sum(tf.multiply(vertex_weights, tf.abs(tf.subtract(vertex_pred, vertex_targets)))), tf.reduce_sum(vertex_weights) )
                
                loss = loss_cls + loss_vertex
            else:
                scores = network.get_output('prob')
                labels = network.get_output('gt_label_2d')
                loss = loss_cross_entropy_single_frame(scores, labels)
    else:
        # classification loss
        scores = network.get_output('outputs')
        labels = network.get_output('labels_gt_2d')
        loss = loss_cross_entropy(scores, labels)

    # optimizer
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
    momentum = cfg.TRAIN.MOMENTUM
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=global_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)

        # thread to load data
        coord = tf.train.Coordinator()
        if cfg.TRAIN.VISUALIZE:
            load_and_enqueue(sess, network, roidb, imdb.num_classes, coord)
        else:
            t = threading.Thread(target=load_and_enqueue, args=(sess, network, roidb, imdb.num_classes, coord))
            t.start()

        print 'Solving...'
        if cfg.TRAIN.VERTEX_REG:
            sw.train_model_vertex(sess, train_op, loss, loss_cls, loss_vertex, learning_rate, max_iters)
        else:
            sw.train_model(sess, train_op, loss, learning_rate, max_iters)
        print 'done solving'

        sess.run(network.close_queue_op)
        coord.request_stop()
        coord.join([t])

# train GAN
def train_gan(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a GAN."""

    # define losses

    # classification loss for the discriminator
    outputs_d = network.get_output('outputs_d')
    scores_d_true = outputs_d[1]
    scores_d_false = outputs_d[0]

    loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(scores_d_true, tf.ones_like(scores_d_true)))
    loss_false = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(scores_d_false, tf.zeros_like(scores_d_false)))
    loss_d = loss_true + loss_false

    # loss for the generator
    output_g = network.get_output('output_g')
    data_gt = network.get_output('data_gt')
    loss_l1 = tf.reduce_mean(tf.abs(tf.sub(output_g, data_gt)))
    loss_ad = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(scores_d_false, tf.ones_like(scores_d_false)))
    loss_g = 0 * loss_l1 + 0.2 * loss_ad

    # optimizer
    train_op_d = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE, cfg.TRAIN.MOMENTUM).minimize(loss_d)
    train_op_g = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE, cfg.TRAIN.MOMENTUM).minimize(loss_g)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)

        # thread to load data
        coord = tf.train.Coordinator()
        if cfg.TRAIN.VISUALIZE:
            load_and_enqueue(sess, network, roidb, imdb.num_classes, coord)
        else:
            t = threading.Thread(target=load_and_enqueue, args=(sess, network, roidb, imdb.num_classes, coord))
            t.start()

        print 'Solving...'
        sw.train_gan(sess, train_op_d, train_op_g, loss_d, loss_true, loss_false, loss_g, loss_l1, loss_ad, max_iters)
        print 'done solving'

        sess.run(network.close_queue_op)
        coord.request_stop()
        coord.join([t])
