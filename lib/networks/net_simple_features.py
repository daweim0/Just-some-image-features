import tensorflow as tf
from networks.network import Network
from fcn.config import cfg

zero_out_module = tf.load_op_library('lib/triplet_flow_loss/triplet_flow_loss.so')


class custom_network(Network):
    def __init__(self):
        self.inputs = cfg.INPUT
        # self.input_format = input_format
        self.num_output_dimensions = 2  # formerly num_classes
        self.num_units = cfg.TRAIN.NUM_UNITS
        self.scale = 1 / cfg.TRAIN.SCALES_BASE[0]
        self.vertex_reg = cfg.TRAIN.VERTEX_REG

        self.data_left = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.data_right = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.gt_flow = tf.placeholder(tf.float32, shape=[None, None, None, self.num_output_dimensions])
        self.occluded = tf.placeholder(tf.int32, shape=[None, None, None, 1])
        self.keep_prob = tf.placeholder(tf.float32)

        # define a queue
        self.q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.float32, tf.int32, tf.float32])
        self.enqueue_op = self.q.enqueue([self.data_left, self.data_right, self.gt_flow, self.occluded, self.keep_prob])
        data_left, data_right, gt_flow, occluded, self.keep_prob_queue = self.q.dequeue()
        self.layers = dict({'data_left': data_left, 'data_right': data_right, 'gt_flow': gt_flow, 'occluded': occluded})

        self.close_queue_op = self.q.close(cancel_pending_enqueues=True)
        self.queue_size_op = self.q.size('queue_size')
        self.trainable = cfg.TRAIN.TRAINABLE
        self.setup()

    def setup(self):
        trainable = True
        reuse = True
#--weights data/imagenet_models/vgg16_convs.npy
        # left tower
        (self.feed('data_left')
         .add_immediate(tf.constant(0.0, tf.float32), name='data_left_tap')
         .conv(3, 3, 64, 1, 1, name='conv1', c_i=3, trainable=trainable, elu=True)
         .conv(3, 3, 64, 1, 1, name='conv2', c_i=64, trainable=trainable, elu=True)
         .conv(3, 3, 64, 1, 1, name='conv3', c_i=64, trainable=trainable, elu=True)
         .conv(3, 3, 64, 1, 1, name='conv4', c_i=64, trainable=trainable, relu=False)
         .add_immediate(tf.constant(0.0, tf.float32), name='upscore_l'))

        # right tower
        (self.feed('data_right')
         .add_immediate(tf.constant(0.0, tf.float32), name='data_right_tap')
         .conv(3, 3, 64, 1, 1, name='conv1', c_i=3, trainable=trainable, elu=True, reuse=True)
         .conv(3, 3, 64, 1, 1, name='conv2', c_i=64, trainable=trainable, elu=True, reuse=True)
         .conv(3, 3, 64, 1, 1, name='conv3', c_i=64, trainable=trainable, elu=True, reuse=True)
         .conv(3, 3, 64, 1, 1, name='conv4', c_i=64, trainable=trainable, relu=False, reuse=True)
         .add_immediate(tf.constant(0.0, tf.float32), name='upscore_r'))

        # triplet loss
        (self.feed(['upscore_l', 'upscore_r', 'gt_flow', 'occluded'])
         .triplet_flow_loss(margin=1, name="triplet_flow_loss_name"))

        (self.feed('gt_flow')
         .avg_pool(2, 2, 2, 2, name='flow_pool1')
         .div_immediate(tf.constant(2.0, tf.float32), name='flow_pool1_out')
         .avg_pool(2, 2, 2, 2, name='flow_pool2')
         .div_immediate(tf.constant(2.0, tf.float32), name='flow_pool2_out')
         .avg_pool(2, 2, 2, 2, name='flow_pool3')
         .div_immediate(tf.constant(2.0, tf.float32), name='flow_pool3_out')
         .deconv(4, 4, 2, 2, 2, name='flow_upscore_4', trainable=False, c_i=2)
         .mult_immediate(tf.constant(2.0, tf.float32), name='flow_pool1_out')
         .deconv(int(16 * self.scale), int(16 * self.scale), 2, int(8 * self.scale), int(8 * self.scale),
                 name='flow_upscore', trainable=False, c_i=2)
         .mult_immediate(tf.constant(8.0, tf.float32), name='flow_pool_out'))

        (self.feed('occluded')
         .cast(tf.float32)
         .max_pool(2, 2, 2, 2, name='occluded_pool1')
         .max_pool(2, 2, 2, 2, name='occluded_pool2')
         .max_pool(2, 2, 2, 2, name='occluded_pool3')
         .max_pool(2, 2, 2, 2, name='occluded_pool4'))
        pass


