import tensorflow as tf
from tensorflow.python.framework import ops
import triplet_flow_loss_op


@ops.RegisterGradient("TripletFlow")
def _triplet_flow_grad(op, input_grad, input_grad_left, input_grad_right):
    # The gradient outputs of the actual operation should not be used as operation outputs. input_grad_left and
    # input_grad_right are the incoming gradients with respect to them, and so they should be ignored.

    diff_left = op.outputs[1]
    diff_right = op.outputs[2]
    margin = op.get_attr('margin')

    # compute gradient
    data_left_grad, data_right_grad = triplet_flow_loss_op.triplet_flow_loss_grad(diff_left, diff_right, input_grad, margin)

    return [data_left_grad, data_right_grad, None, None, None, None]  # List of one Tensor, since we have three input
