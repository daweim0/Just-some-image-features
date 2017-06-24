import tensorflow as tf
import os.path as osp


filename = osp.join(osp.dirname(__file__), 'triplet_flow_loss.so')
_triplet_flow_loss_module = tf.load_op_library(filename)
triplet_flow_loss = _triplet_flow_loss_module.triplet_flow
triplet_flow_loss_grad = _triplet_flow_loss_module.triplet_flow_grad
