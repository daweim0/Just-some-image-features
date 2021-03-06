import tensorflow as tf
from tensorflow.python.framework import ops
import triplet_loss_op


@ops.RegisterGradient("TripletFlow")
def _triplet_grad(op, grad, _):

  diff = op.outputs[1]
  margin = op.get_attr('margin')

  # compute gradient
  data_left_grad, data_right_grad = triplet_loss_op.triplet_loss_grad(diff, grad, margin)

  return [data_left_grad, None, None]  # List of one Tensor, since we have three input
