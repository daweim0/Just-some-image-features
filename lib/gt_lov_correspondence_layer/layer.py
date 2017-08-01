# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""The data layer used during training to train a FCN for single frames.
"""

from fcn.config import cfg
import numpy as np
from utils.voxelizer import Voxelizer
import random

# import pyximport; pyximport.install()
from gt_lov_correspondence_layer.minibatch import get_minibatch


class GtLOVFlowDataLayer(object):
    """FCN data layer used for training."""

    def __init__(self, roidb, num_classes, single=False):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        self._voxelizer = Voxelizer(cfg.TRAIN.GRID_SIZE, num_classes)
        self._shuffle_roidb_inds()
        if single:
            self._imgs_per_batch = 1
        else:
            self._imgs_per_batch = cfg.TRAIN.IMS_PER_BATCH
            # preload_data(self._roidb)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + self._imgs_per_batch >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self._imgs_per_batch]
        self._cur += self._imgs_per_batch

        r_img = db_inds.copy()
        roidb_len = len(self._roidb)
        # for i in xrange(len(self._roidb)):
        for i in xrange(len(db_inds)):
            # rand = random.randint(0, roidb_len)
            rand = 50
            for k in xrange(roidb_len):
                new_index = (db_inds[i] + rand + k) % roidb_len
                if self._roidb[db_inds[i]]['video_id'].split(" ")[0] == self._roidb[new_index]['video_id'].split(" ")[0]:
                    r_img[i] = new_index
                    break

        db_inds_pairs = np.vstack([db_inds, r_img]).transpose().flatten()

        return db_inds_pairs

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        db_inds = self._get_next_minibatch_inds()
        # minibatch_db = [self._roidb[i] for i in db_inds]
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._voxelizer)

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        return blobs