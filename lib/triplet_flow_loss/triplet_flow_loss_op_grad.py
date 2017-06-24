import tensorflow as tf
from tensorflow.python.framework import ops
import triplet_flow_loss_op


def test():
  print "hi"


@ops.RegisterGradient("TripletFlow")
def _triplet_flow_grad(op, grad, grad_right, _):

  diff_left = op.outputs[1]
  diff_right = op.outputs[1]
  margin = op.get_attr('margin')

  # compute gradient
  data_left_grad, data_right_grad = triplet_flow_loss_op.triplet_flow_loss_grad(diff_left, diff_right, grad, margin)

  return [data_left_grad, data_right_grad, None, None]  # List of one Tensor, since we have three input
