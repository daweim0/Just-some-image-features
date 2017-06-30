import tensorflow as tf
from networks.network import Network
from fcn.config import cfg
zero_out_module = tf.load_op_library('lib/triplet_flow_loss/triplet_flow_loss.so')


class vgg16_flow_features(Network):
    def __init__(self, input_format, num_classes, num_units, scales, vertex_reg=False, trainable=True):
        self.inputs = []
        self.input_format = input_format
        self.num_output_dimensions = 2  # formerly num_classes
        self.num_units = num_units
        self.scale = 1 / scales[0]
        self.vertex_reg = vertex_reg

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
        self.trainable = trainable
        self.setup()

    def setup(self):
        trainable = False
        reuse = True

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

        (self.feed('conv5_3_l')
         .conv(1, 1, self.num_units, 1, 1, name='score_conv5', c_i=512, )
         .deconv(4, 4, self.num_units, 2, 2, name='upscore_conv5_l', trainable=False))

        (self.feed('conv4_3_l')
         .conv(1, 1, self.num_units, 1, 1, name='score_conv4', c_i=512, )
         .add_immediate(tf.constant(0.0, tf.float32), name='score_conv4_l'))

        (self.feed('conv3_l')
         .conv(1, 1, self.num_units, 1, 1, name='score_conv3', c_i=256, )
         .add_immediate(tf.constant(0.0, tf.float32), name='score_conv3_l'))

        (self.feed('conv2_l')
         .conv(1, 1, self.num_units, 1, 1, name='score_conv2', c_i=128, )
         .add_immediate(tf.constant(0.0, tf.float32), name='score_conv2_l'))

        (self.feed('conv1_l')
         .conv(1, 1, self.num_units, 1, 1, name='score_conv1', c_i=64, )
         .add_immediate(tf.constant(0.0, tf.float32), name='score_conv1_l'))

        (self.feed('score_conv4_l', 'upscore_conv5_l')
         .add(name='add_score_l')
         .dropout(self.keep_prob_queue, name='dropout_l')
         .deconv(4, 4, self.num_units, 2, 2, name='upscore_conv4_l', trainable=False))

        if cfg.NET_CONF.CONV3_SKIP_LINK:
            self.feed('upscore_conv4_l', 'score_conv3_l')
            self.add(name='add_score_3_l')
        else:
            self.feed('upscore_conv4_l')
        self.deconv(4, 4, self.num_units, 2, 2, name='deconv3_l', trainable=False, c_i=self.num_units)

        if cfg.NET_CONF.CONV2_SKIP_LINK:
            self.feed('deconv3_l', 'score_conv2_l')
            self.add(name='add_score_2_l')
        else:
            self.feed('deconv3_l')
        self.deconv(4, 4, self.num_units, 2, 2, name='deconv2_l', trainable=False, c_i=self.num_units)

        if cfg.NET_CONF.CONV1_SKIP_LINK:
            self.feed('deconv2_l', 'score_conv1_l')
            self.add(name='add_score_1_l')
        else:
            self.feed('deconv2_l')
        self.add_immediate(tf.constant(0.0, tf.float32), name='upscore_l')

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

        (self.feed('conv5_3_r')
             .conv(1, 1, self.num_units, 1, 1, name='score_conv5', c_i=512, reuse=reuse)
             .deconv(4, 4, self.num_units, 2, 2, name='upscore_conv5_r', trainable=False))

        (self.feed('conv4_3_r')
             .conv(1, 1, self.num_units, 1, 1, name='score_conv4', c_i=512, reuse=reuse)
             .add_immediate(tf.constant(0.0, tf.float32), name='score_conv4_r'))

        (self.feed('conv3_r')
             .conv(1, 1, self.num_units, 1, 1, name='score_conv3', c_i=256, reuse=reuse)
             .add_immediate(tf.constant(0.0, tf.float32), name='score_conv3_r'))

        (self.feed('conv2_r')
             .conv(1, 1, self.num_units, 1, 1, name='score_conv2', c_i=128, reuse=reuse)
             .add_immediate(tf.constant(0.0, tf.float32), name='score_conv2_r'))

        (self.feed('conv1_r')
             .conv(1, 1, self.num_units, 1, 1, name='score_conv1', c_i=64, reuse=reuse)
             .add_immediate(tf.constant(0.0, tf.float32), name='score_conv1_r'))

        (self.feed('score_conv4_r', 'upscore_conv5_r')
         .add(name='add_score_r')
         .dropout(self.keep_prob_queue, name='dropout_r_1')
         .deconv(4, 4, self.num_units, 2, 2, name='upscore_conv4_r', trainable=False))

        if cfg.NET_CONF.CONV3_SKIP_LINK:
            self.feed('upscore_conv4_r', 'score_conv3_r')
            self.add(name='add_score_3_r')
        else:
            self.feed('upscore_conv4_r')
            self.add_immediate(tf.constant(0.0, tf.float32), name='add_score_3_r')
        self.deconv(4, 4, self.num_units, 2, 2, name='deconv3_r', trainable=False, c_i=self.num_units)
        
        if cfg.NET_CONF.CONV2_SKIP_LINK:
            self.feed('deconv3_r', 'score_conv2_r')
            self.add(name='add_score_2_r')
        else:
            self.feed('deconv3_r')
        self.deconv(4, 4, self.num_units, 2, 2, name='deconv2_r', trainable=False, c_i=self.num_units)
        
        if cfg.NET_CONF.CONV1_SKIP_LINK:
            self.feed('deconv2_r', 'score_conv1_r')
            self.add(name='add_score_1_r')
        else:
            self.feed('deconv2_r')
        self.add_immediate(tf.constant(0.0, tf.float32), name='upscore_r')

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


