__author__ = 'yuxiang'

import os
import datasets
import datasets.imdb
import cPickle
import numpy as np
import cv2
from fcn.config import cfg

class lov_synthetic(datasets.imdb):
    def __init__(self, image_set, lov_path = None):
        datasets.imdb.__init__(self, 'lov_synthetic_' + image_set)
        self._image_set = image_set
        self._lov_synthetic_path = self._get_default_path() + "/" + image_set if lov_path is None \
                            else lov_path
        self._data_path = os.path.join(self._lov_synthetic_path)

        self._classes = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')

        self._blacklist = {}

        # # only train on cracker box
        # self._blacklist = {'002_master_chef_can', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
        #                  '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
        #                  '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
        #                  '051_large_clamp', '052_extra_large_clamp', '061_foam_brick'}

        # # only train on the drill and cracker box
        # self._blacklist = {'002_master_chef_can', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
        #                  '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
        #                  '021_bleach_cleanser', '024_bowl', '025_mug', '036_wood_block', '037_scissors', '040_large_marker', \
        #                  '051_large_clamp', '052_extra_large_clamp', '061_foam_brick'}

        # standard balcklist
        self._blacklist = {'007_tuna_fish_can', '011_banana', '019_pitcher_base', '024_bowl', '025_mug',
                          '036_wood_block', '037_scissors', '051_large_clamp', '052_extra_large_clamp',
                          '061_foam_brick'}

        # self._blacklist = {'004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
        #                  '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
        #                  '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
        #                  '051_large_clamp', '052_extra_large_clamp', '061_foam_brick'}


        self._class_colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
                              (192, 0, 0), (0, 192, 0), (0, 0, 192)]

        self._class_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # self._extents = self._load_object_extents()

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        self.background_imgs = [os.listdir("data/backgrounds/")]

        assert os.path.exists(self._lov_synthetic_path), \
                'lov path does not exist: {}'.format(self._lov_synthetic_path)
        assert os.path.exists(self._data_path), \
                'Data path does not exist: {}'.format(self._data_path)

    # image
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """

        image_path = os.path.join(self._data_path, index + '_color' + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # depth
    def depth_path_at(self, i):
        """
        Return the absolute path to depth i in the image sequence.
        """
        return self.depth_path_from_index(self.image_index[i])

    def depth_path_from_index(self, index):
        """
        Construct an depth path from the image's "index" identifier.
        """
        depth_path = os.path.join(self._data_path, index + '_depth' + self._image_ext)
        assert os.path.exists(depth_path), \
                'Path does not exist: {}'.format(depth_path)
        return depth_path

    # camera pose
    def pose_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.pose_path_from_index(self.image_index[i])

    def pose_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        metadata_path = os.path.join(self._data_path, index + '_pose.txt')
        assert os.path.exists(metadata_path), \
                'Path does not exist: {}'.format(metadata_path)
        return metadata_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        if cfg.SET_VARIANT == "":
            image_set_file = os.path.join(self._lov_synthetic_path, self._image_set + cfg.TRAIN.IMAGE_LIST_NAME +
                                          cfg.SET_VARIANT + '.txt')
        else:
            image_set_file = os.path.join(self._lov_synthetic_path, self._image_set + cfg.TRAIN.IMAGE_LIST_NAME +
                                          "-" + cfg.SET_VARIANT + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.rstrip('\n').split(" ") for x in f]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where KITTI is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'lov_synthetic')


    def _load_object_extents(self):

        extent_file = os.path.join(self._lov_synthetic_path, 'extents.txt')
        assert os.path.exists(extent_file), \
                'Path does not exist: {}'.format(extent_file)

        extents = np.zeros((self.num_classes, 3), dtype=np.float32)
        extents[1:, :] = np.loadtxt(extent_file)

        return extents


    def compute_class_weights(self):

        print 'computing class weights'
        num_classes = self.num_classes
        count = np.zeros((num_classes,), dtype=np.int64)
        for index in self.image_index:
            # label path
            label_path = self.label_path_from_index(index)
            im = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            for i in xrange(num_classes):
                I = np.where(im == i)
                count[i] += len(I[0])

        for i in xrange(num_classes):
            self._class_weights[i] = min(float(count[0]) / float(count[i]), 10.0)
            print self._classes[i], self._class_weights[i]


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        # cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = cPickle.load(fid)
        #     print '{} gt roidb loaded from {}'.format(self.name, cache_file)
        #     # print 'class weights: ', roidb[0]['class_weights']
        #     return roidb

        # self.compute_class_weights()

        gt_roidb = [self._load_lov_annotation(index) for index in self.image_index if index[0].split("/")[0] not in self._blacklist]

        # with open(cache_file, 'wb') as fid:
        #     cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        # print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def _load_lov_annotation(self, index):
        """
        Load class name and meta data
        """
        # image path
        image_path_l = self.image_path_from_index(index[0])

        image_path_r = self.image_path_from_index(index[1])

        # depth path
        depth_path_l = self.depth_path_from_index(index[0])
        depth_path_r = self.depth_path_from_index(index[1])

        # metadata path
        pose_path_l = self.pose_path_from_index(index[0])
        pose_path_r = self.pose_path_from_index(index[1])

        # parse image name
        video_id = index[0].split("/")[0]
        
        return {'image': image_path_l,
                'image_right': image_path_r,
                'depth': depth_path_l,
                'depth_right': depth_path_r,
                'pose': pose_path_l,
                'pose_right': pose_path_r,
                'video_id': video_id,
                'flipped': False}

    def _process_label_image(self, label_image):
        """
        change label image to label index
        """
        class_colors = self._class_colors
        width = label_image.shape[1]
        height = label_image.shape[0]
        label_index = np.zeros((height, width), dtype=np.float32)

        # label image is in BGR order
        index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
        for i in xrange(len(class_colors)):
            color = class_colors[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            label_index[I] = i

        return label_index


    def labels_to_image(self, im, labels):
        class_colors = self._class_colors
        height = labels.shape[0]
        width = labels.shape[1]
        image_r = np.zeros((height, width), dtype=np.float32)
        image_g = np.zeros((height, width), dtype=np.float32)
        image_b = np.zeros((height, width), dtype=np.float32)

        for i in xrange(len(class_colors)):
            color = class_colors[i]
            I = np.where(labels == i)
            image_r[I] = color[0]
            image_g[I] = color[1]
            image_b[I] = color[2]

        image = np.stack((image_r, image_g, image_b), axis=-1)

        return image.astype(np.uint8)


    def evaluate_segmentations(self, segmentations, output_dir):
        print 'evaluating segmentations'
        # compute histogram
        n_cl = self.num_classes
        hist = np.zeros((n_cl, n_cl))

        # make image dir
        image_dir = os.path.join(output_dir, 'images')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # make matlab result dir
        import scipy.io
        mat_dir = os.path.join(output_dir, 'mat')
        if not os.path.exists(mat_dir):
            os.makedirs(mat_dir)

        # for each image
        for im_ind, index in enumerate(self.image_index):
            # read ground truth labels
            im = cv2.imread(self.label_path_from_index(index), cv2.IMREAD_UNCHANGED)
            gt_labels = im.astype(np.float32)

            # predicated labels
            sg_labels = segmentations[im_ind]['labels']

            hist += self.fast_hist(gt_labels.flatten(), sg_labels.flatten(), n_cl)

            '''
            # label image
            rgba = cv2.imread(self.image_path_from_index(index), cv2.IMREAD_UNCHANGED)
            image = rgba[:,:,:3]
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            image[I[0], I[1], :] = 255
            label_image = self.labels_to_image(image, sg_labels)

            # save image
            filename = os.path.join(image_dir, '%04d.png' % im_ind)
            print filename
            cv2.imwrite(filename, label_image)
            '''
            '''
            # save matlab result
            labels = {'labels': sg_labels}
            filename = os.path.join(mat_dir, '%04d.mat' % im_ind)
            print filename
            scipy.io.savemat(filename, labels)
            #'''

        # overall accuracy
        acc = np.diag(hist).sum() / hist.sum()
        print 'overall accuracy', acc
        # per-class accuracy
        acc = np.diag(hist) / hist.sum(1)
        print 'mean accuracy', np.nanmean(acc)
        # per-class IU
        print 'per-class IU'
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        for i in range(n_cl):
            print '{} {}'.format(self._classes[i], iu[i])
        print 'mean IU', np.nanmean(iu)
        freq = hist.sum(1) / hist.sum()
        print 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum()

        filename = os.path.join(output_dir, 'segmentation.txt')
        with open(filename, 'wt') as f:
            for i in range(n_cl):
                f.write('{:f}\n'.format(iu[i]))

        filename = os.path.join(output_dir, 'confusion_matrix.txt')
        with open(filename, 'wt') as f:
            for i in range(n_cl):
                for j in range(n_cl):
                    f.write('{:f} '.format(hist[i, j]))
                f.write('\n')


if __name__ == '__main__':
    d = datasets.lov('train')
    res = d.roidb
    from IPython import embed; embed()




import os
file = open('train.txt', 'w+')
for folder in os.listdir('.'):
    try:
         n_files = len([name for name in os.listdir(folder)]) / 3
         for i in range(n_files):
                 file.write(folder + "/" + str(i).zfill(6) + "\n")
    except:
         pass


for folder in os.listdir('.'):
    try:
        for cur_file in os.listdir(folder):
             if cur_file.count('color') != 0:
                     file.write(folder + "/" + cur_file.split("_")[0] + "\n")
    except:
        pass
