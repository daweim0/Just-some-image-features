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

        if cfg.NET_CONF.CONV1_SKIP_LINK:
            self.skip_1_mult = tf.constant(1.0, tf.float32)
        else:
            self.skip_1_mult = tf.constant(0.0, tf.float32)
        if cfg.NET_CONF.CONV2_SKIP_LINK:
            self.skip_2_mult = tf.constant(1.0, tf.float32)
        else:
            self.skip_2_mult = tf.constant(0.0, tf.float32)
        if cfg.NET_CONF.CONV3_SKIP_LINK:
            self.skip_4_mult = tf.constant(1.0, tf.float32)
        else:
            self.skip_4_mult = tf.constant(0.0, tf.float32)

        self.setup()

    def setup(self):
        trainable = self.trainable
        reuse = True

        # scaled versions of ground truth
        (self.feed('gt_flow')
         .avg_pool(2, 2, 2, 2, name='flow_pool1')
         .div_immediate(tf.constant(2.0, tf.float32), name='gt_flow_2x')
         .avg_pool(2, 2, 2, 2, name='flow_pool2')
         .div_immediate(tf.constant(2.0, tf.float32), name='gt_flow_4x')
         .avg_pool(2, 2, 2, 2, name='flow_pool3')
         .div_immediate(tf.constant(2.0, tf.float32), name='gt_flow_8x')
         .avg_pool(2, 2, 2, 2, name='flow_pool4')
         .div_immediate(tf.constant(2.0, tf.float32), name='gt_flow_16x'))

        (self.feed('occluded').cast(tf.float32)
         .max_pool_int(2, 2, 2, 2, name='occluded_2x').cast(tf.float32)
         .max_pool_int(2, 2, 2, 2, name='occluded_4x').cast(tf.float32)
         .max_pool_int(2, 2, 2, 2, name='occluded_8x').cast(tf.float32)
         .max_pool_int(2, 2, 2, 2, name='occluded_16x'))

        # left tower
        (self.feed('data_left')
         .add_immediate(tf.constant(0.0, tf.float32), name='data_left_tap')
         .conv(3, 3, 64, 1, 1, name='conv1_1', c_i=3, trainable=trainable)
         .conv(3, 3, 64, 1, 1, name='conv1_2', c_i=64, trainable=trainable)
         .add_immediate(tf.constant(0.0, tf.float32), name='conv1_l')
         .max_pool(2, 2, 2, 2, name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1', c_i=64, trainable=trainable)
         .conv(3, 3, 128, 1, 1, name='conv2_2', c_i=128, trainable=trainable)
         .add_immediate(tf.constant(0.0, tf.float32), name='conv2_l')
         .max_pool(2, 2, 2, 2, name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1', c_i=128, trainable=trainable)
         .conv(3, 3, 256, 1, 1, name='conv3_2', c_i=256, trainable=trainable)
         .conv(3, 3, 256, 1, 1, name='conv3_3', c_i=256, trainable=trainable)
         .add_immediate(tf.constant(0.0, tf.float32), name='conv3_l')
         .max_pool(2, 2, 2, 2, name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1', c_i=256, trainable=trainable)
         .conv(3, 3, 512, 1, 1, name='conv4_2', c_i=512, trainable=trainable)
         .conv(3, 3, 512, 1, 1, name='conv4_3', c_i=512, trainable=trainable)
         .add_immediate(tf.constant(0.0, tf.float32), name='conv4_3_l')
         .max_pool(2, 2, 2, 2, name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1', c_i=512, trainable=trainable)
         .conv(3, 3, 512, 1, 1, name='conv5_2', c_i=512, trainable=trainable)
         .conv(3, 3, 512, 1, 1, name='conv5_3', c_i=512, trainable=trainable)
         .add_immediate(tf.constant(0.0, tf.float32), name='conv5_3_l'))

        # 16x scaling input
        (self.feed('conv5_3_l')
         .conv(1, 1, 128, 1, 1, name='16_cov_1', c_i=512, elu=True)
         .conv(1, 1, 128, 1, 1, name='16_cov_2', c_i=128, elu=True)
         .add_immediate(tf.constant(0.0, tf.float32), name='features_16x_l')
         .deconv(4, 4, 128, 2, 2, name='upscale_16x_l', trainable=False))

        # 8x scaling input
        (self.feed('conv4_3_l')
         .conv(1, 1, 128, 1, 1, name='8x_skip_cov_1', c_i=512, elu=True)
         .add_immediate(tf.constant(0.0, tf.float32), name='skip_link_8x_l'))
        (self.feed('upscale_16x_l', 'skip_link_8x_l')
         .concat(axis=3, name='8_concat')
         .conv(1, 1, 128, 1, 1, name='8_cov_1', c_i=2 * 128, elu=True)
         .conv(1, 1, 128, 1, 1, name='8_cov_2', c_i=128, elu=True)
         .add_immediate(tf.constant(0.0, tf.float32), name='features_8x_l')
         .deconv(4, 4, 128, 2, 2, name='upscale_8x_l', trainable=False))

        # 4x scaling input
        (self.feed('conv3_l')
         .conv(1, 1, 64, 1, 1, name='4x_skip_conv_1', c_i=256)
         .mult_immediate(self.skip_4_mult, name='skip_link_4x_l'))
        (self.feed('upscale_8x_l', 'skip_link_4x_l')
         .concat(axis=3, name='4_concat')
         .conv(1, 1, 128, 1, 1, name='4_cov_1', c_i=128 + 64, elu=True)
         .conv(1, 1, 64, 1, 1, name='4_cov_2', c_i=128, elu=True)
         .add_immediate(tf.constant(0.0, tf.float32), name='features_4x_l')
         .deconv(4, 4, 64, 2, 2, name='upscale_4x_l', trainable=False))

        # 2x scaling input
        (self.feed('conv2_l')
         .conv(1, 1, 64, 1, 1, name='2x_skip_conv_1', c_i=128)
         .mult_immediate(self.skip_4_mult, name='skip_link_2x_l'))
        (self.feed('upscale_4x_l', 'skip_link_2x_l')
         .concat(axis=3, name='2_concat')
         .conv(1, 1, 2 * 64, 1, 1, name='2_cov_1', c_i=2 * 64, elu=True)
         .conv(1, 1, 64, 1, 1, name='2_cov_2', c_i=2 * 64, elu=True)
         .add_immediate(tf.constant(0.0, tf.float32), name='features_2x_l')
         .deconv(4, 4, 64, 2, 2, name='upscale_2x_l', trainable=False))

        # 1x scaling input
        (self.feed('conv1_l')
         .conv(1, 1, 64, 1, 1, name='1x_skip_conv_1', c_i=64)
         .mult_immediate(self.skip_1_mult, name='skip_link_1x_l'))
        (self.feed('upscale_2x_l', 'skip_link_1x_l')
         .concat(axis=3, name='1_concat')
         .conv(1, 1, 64, 1, 1, name='1_cov_1', c_i=2 * 64, elu=True)
         .conv(1, 1, 64, 1, 1, name='1_cov_2', c_i=64, elu=True)
         .add_immediate(tf.constant(0.0, tf.float32), name='features_1x_l'))

        # right tower
        (self.feed('data_right')
         .add_immediate(tf.constant(0.0, tf.float32), name='data_right_tap')
         .conv(3, 3, 64, 1, 1, name='conv1_1', c_i=3, trainable=trainable, reuse=reuse)
         .conv(3, 3, 64, 1, 1, name='conv1_2', c_i=64, trainable=trainable, reuse=reuse)
         .add_immediate(tf.constant(0.0, tf.float32), name='conv1_r')
         .max_pool(2, 2, 2, 2, name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1', c_i=64, trainable=trainable, reuse=reuse)
         .conv(3, 3, 128, 1, 1, name='conv2_2', c_i=128, trainable=trainable, reuse=reuse)
         .add_immediate(tf.constant(0.0, tf.float32), name='conv2_r')
         .max_pool(2, 2, 2, 2, name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1', c_i=128, trainable=trainable, reuse=reuse)
         .conv(3, 3, 256, 1, 1, name='conv3_2', c_i=256, trainable=trainable, reuse=reuse)
         .conv(3, 3, 256, 1, 1, name='conv3_3', c_i=256, trainable=trainable, reuse=reuse)
         .add_immediate(tf.constant(0.0, tf.float32), name='conv3_r')
         .max_pool(2, 2, 2, 2, name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1', c_i=256, trainable=trainable, reuse=reuse)
         .conv(3, 3, 512, 1, 1, name='conv4_2', c_i=512, trainable=trainable, reuse=reuse)
         .conv(3, 3, 512, 1, 1, name='conv4_3', c_i=512, trainable=trainable, reuse=reuse)
         .add_immediate(tf.constant(0.0, tf.float32), name='conv4_3_r')
         .max_pool(2, 2, 2, 2, name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1', c_i=512, trainable=trainable, reuse=reuse)
         .conv(3, 3, 512, 1, 1, name='conv5_2', c_i=512, trainable=trainable, reuse=reuse)
         .conv(3, 3, 512, 1, 1, name='conv5_3', c_i=512, trainable=trainable, reuse=reuse)
         .add_immediate(tf.constant(0.0, tf.float32), name='conv5_3_r'))

        # 16x scaling input
        (self.feed('conv5_3_r')
         .conv(1, 1, 128, 1, 1, name='16_cov_1', c_i=512, elu=True, reuse=reuse)
         .conv(1, 1, 128, 1, 1, name='16_cov_2', c_i=128, elu=True, reuse=reuse)
         .add_immediate(tf.constant(0.0, tf.float32), name='features_16x_r')
         .deconv(4, 4, 128, 2, 2, name='upscale_16x_r', trainable=False))

        # 8x scaling input
        (self.feed('conv4_3_r')
         .conv(1, 1, 128, 1, 1, name='8x_skip_cov_1', c_i=512, elu=True, reuse=reuse)
         .add_immediate(tf.constant(0.0, tf.float32), name='skip_link_8x_r'))
        (self.feed('upscale_16x_r', 'skip_link_8x_r')
         .concat(axis=3, name='8_concat')
         .conv(1, 1, 128, 1, 1, name='8_cov_1', c_i=2*128, elu=True, reuse=reuse)
         .conv(1, 1, 128, 1, 1, name='8_cov_2', c_i=128, elu=True, reuse=reuse)
         .add_immediate(tf.constant(0.0, tf.float32), name='features_8x_r')
         .deconv(4, 4, 128, 2, 2, name='upscale_8x_r', trainable=False))

        # 4x scaling input
        (self.feed('conv3_r')
         .conv(1, 1, 64, 1, 1, name='4x_skip_conv_1', c_i=256, reuse=reuse)
         .mult_immediate(self.skip_4_mult, name='skip_link_4x_r'))
        (self.feed('upscale_8x_r', 'skip_link_4x_r')
         .concat(axis=3, name='4_concat')
         .conv(1, 1, 128, 1, 1, name='4_cov_1', c_i=128 + 64, elu=True, reuse=reuse)
         .conv(1, 1, 64, 1, 1, name='4_cov_2', c_i=128, elu=True, reuse=reuse)
         .add_immediate(tf.constant(0.0, tf.float32), name='features_4x_r')
         .deconv(4, 4, 64, 2, 2, name='upscale_4x_r', trainable=False))

        # 2x scaling input
        (self.feed('conv2_r')
         .conv(1, 1, 64, 1, 1, name='2x_skip_conv_1', c_i=128, reuse=reuse)
         .mult_immediate(self.skip_4_mult, name='skip_link_2x_r'))
        (self.feed('upscale_4x_r', 'skip_link_2x_r')
         .concat(axis=3, name='2_concat')
         .conv(1, 1, 2 * 64, 1, 1, name='2_cov_1', c_i=2 * 64, elu=True, reuse=reuse)
         .conv(1, 1, 64, 1, 1, name='2_cov_2', c_i=2 * 64, elu=True, reuse=reuse)
         .add_immediate(tf.constant(0.0, tf.float32), name='features_2x_r')
         .deconv(4, 4, 64, 2, 2, name='upscale_2x_r', trainable=False))

        # 1x scaling input
        (self.feed('conv1_r')
         .conv(1, 1, 64, 1, 1, name='1x_skip_conv_1', c_i=64, reuse=reuse)
         .mult_immediate(self.skip_1_mult, name='skip_link_1x_r'))
        (self.feed('upscale_2x_r', 'skip_link_1x_r')
         .concat(axis=3, name='1_concat')
         .conv(1, 1, 64, 1, 1, name='1_cov_1', c_i=2 * 64, elu=True, reuse=reuse)
         .conv(1, 1, 64, 1, 1, name='1_cov_2', c_i=64, elu=True, reuse=reuse)
         .add_immediate(tf.constant(0.0, tf.float32), name='features_1x_r'))


        # triplet loss
        (self.feed(['features_16x_l', 'features_16x_r', 'gt_flow_16x', 'occluded_16x'])
         .triplet_flow_loss(margin=1.0, negative_radius=cfg.NET_CONF.NEGATIVE_RADIUS, name="triplet_loss_16x"))

        (self.feed(['features_4x_l', 'features_4x_r', 'gt_flow_4x', 'occluded_4x'])
         .triplet_flow_loss(margin=1.0, negative_radius=cfg.NET_CONF.NEGATIVE_RADIUS, name="triplet_loss_4x"))

        (self.feed(['features_1x_l', 'features_1x_r', 'gt_flow', 'occluded'])
         .triplet_flow_loss(margin=1.0, negative_radius=cfg.NET_CONF.NEGATIVE_RADIUS, name="triplet_loss_1x"))

        final_output = (self.get_output('triplet_loss_16x')[0] + self.get_output('triplet_loss_4x')[0] + self.get_output('triplet_loss_1x')[0]) / 3.0
        self.layers["final_triplet_loss"] = [final_output]

        pass


