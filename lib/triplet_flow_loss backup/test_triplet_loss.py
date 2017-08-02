# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Relu and ReluGrad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class EluTest(test.TestCase):

  def _npElu(self, np_features):
    return np.where(np_features < 0, np.exp(np_features) - 1, np_features)

  def testNpElu(self):
    self.assertAllClose(
        np.array([[-0.59343034025, 0.7, -0.39346934028, 0.3, -0.09516258196],
                  [0.1, -0.25918177931, 0.5, -0.5034146962, 0.9]]),
        self._npElu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7, 0.9]
                     ])))

  def _testElu(self, np_features, use_gpu=False):
    np_elu = self._npElu(np_features)
    with self.test_session(use_gpu=use_gpu):
      elu = nn_ops.elu(np_features)
      tf_elu = elu.eval()
    self.assertAllClose(np_elu, tf_elu)
    self.assertShapeEqual(np_elu, elu)

  def testNumbers(self):
    for t in [np.float16, np.float32, np.float64]:
      self._testElu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      self._testElu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=True)

  def testGradientFloat32(self):
    with self.test_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = constant_op.constant(x_val, name="x")
      y = nn_ops.elu(x, name="elu")
      x_init = np.asarray(x_val, dtype=np.float32, order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    print("elu (float32) gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.test_session():
      x_val = [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]]
      x = constant_op.constant(x_val, dtype=dtypes.float64, name="x")
      y = nn_ops.elu(x, name="elu")
      x_init = np.asarray(x_val, dtype=np.float64, order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    print("elu (float64) gradient err = ", err)
    self.assertLess(err, 1e-6)

  def testGradGradFloat32(self):
    with self.test_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          name="x")
      y = nn_ops.elu(x, name="elu")
      z = gradients_impl.gradients(y, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], z[0], [2, 5], x_init_value=x_init)
    print("elu (float32) gradient of gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    with self.test_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          dtype=dtypes.float64,
          name="x")
      y = nn_ops.elu(x, name="elu")
      z = gradients_impl.gradients(y, x)
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], z[0], [2, 5], x_init_value=x_init)
    print("elu (float64) gradient of gradient err = ", err)
    self.assertLess(err, 1e-6)


if __name__ == "__main__":
  test.main()


