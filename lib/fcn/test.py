# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# Heavily modified by David Michelman
# --------------------------------------------------------

"""Test a FCN on an imdb (image database)."""

from fcn.config import cfg
from utils.blob import im_list_to_blob, pad_im
import numpy as np
import scipy.io
from utils import sintel_utils
from matplotlib import pyplot as plt
import random
import cv2
from gt_flow_data_layer.layer import GtFlowDataLayer
from gt_lov_correspondence_layer.layer import GtLOVFlowDataLayer
from gt_lov_synthetic_layer.layer import GtLOVSyntheticLayer
import scipy.ndimage
import triplet_flow_loss.run_slow_flow_calculator_process

# pyperclip might not be installed and this code shouldn't fail if it isn't
try:
    import pyperclip
except:
    print "pyperclip not imported"
from ast import literal_eval


###################
# test flow
###################
def test_flow_net(sess, net, imdb, weights_filename, save_image=False, calculate_EPE_all_data=True, show_arrows=False):

    roidb_ordering = np.arange(len(imdb.roidb))
    n_images = len(imdb.roidb)
    roidb_ordering = roidb_ordering[0:n_images]
    EPE_list = list()

    if cfg.IMDB_NAME.count("lov_synthetic") != 0:
        data_layer = GtLOVSyntheticLayer(imdb.roidb, None, single=True)
    elif cfg.INPUT == "LEFT_RIGHT_CORRESPONDENCE":
        data_layer = GtLOVFlowDataLayer(imdb.roidb, None, single=True)
    else:
        data_layer = GtFlowDataLayer(imdb.roidb, None, single=True)

    class_epe_set = {}

    global image_index_pos
    image_index_pos = -1
    while image_index_pos < n_images - 1:
        image_index_pos += 1

        # Get network outputs
        blobs = data_layer.forward()
        left_blob = blobs['left_image']
        right_blob = blobs['right_image']
        flow_blob = blobs['flow']
        depth_blob = blobs['depth']
        gt_flow = flow_blob[0]
        occluded_blob = blobs['occluded']
        warped_blob = blobs['warped_im']
        left_labels_blob = blobs['left_labels']
        right_labels_blob = blobs['right_labels']

        if calculate_EPE_all_data == True and blobs['roidb'][0]['video_id'] in class_epe_set and class_epe_set[blobs['roidb'][0]['video_id']][1] > 4:
            continue

        index = roidb_ordering[image_index_pos]
        images = imdb.roidb[index]

        network_inputs = {net.data_left: left_blob, net.data_right: right_blob, net.gt_flow: flow_blob,
                          net.occluded: occluded_blob, net.labels_left: left_labels_blob,
                          net.labels_right: right_labels_blob, net.keep_prob: 1.0}


        network_outputs = [net.get_output('features_1x_l'), net.get_output('features_1x_r'), net.get_output('gt_flow'),
                           net.get_output('final_triplet_loss'), net.get_output('occluded'),
                           net.get_output("features_2x_l"),  net.get_output("features_4x_l"), net.get_output("features_8x_l"),
                           net.get_output("features_2x_r"), net.get_output("features_4x_r"), net.get_output("features_8x_r"),
                           net.get_output("occluded_2x"), net.get_output("occluded_4x"), net.get_output("occluded_8x")]
        results = siphon_outputs_single_frame(sess, net, network_inputs, network_outputs)

        # for FPN style networks
        features_l = results[0][0]
        features_r = results[1][0]

        # # for networks like net_multi_scale_features
        # features_l = np.concatenate([
        #                             scipy.ndimage.zoom(np.squeeze(results[7]), (8, 8, 1), order=1),
        #                             # scipy.ndimage.zoom(np.squeeze(results[6]), (4, 4, 1), order=1),
        #                             # scipy.ndimage.zoom(np.squeeze(results[5]), (2, 2, 1), order=1),
        #                             np.squeeze(results[0])], axis=2)
        #
        # features_r = np.concatenate([
        #                             scipy.ndimage.zoom(np.squeeze(results[10]), (8, 8, 1), order=1),
        #                             # scipy.ndimage.zoom(np.squeeze(results[9]), (4, 4, 1), order=1),
        #                             # scipy.ndimage.zoom(np.squeeze(results[8]), (2, 2, 1), order=1),
        #                             np.squeeze(results[1])], axis=2)

        # single scale flow calculation
        left_pyramid = [features_l]
        right_pyramid = [features_r]
        occluded_pyramid = [np.squeeze(results[4])]
        # occluded_pyramid = [np.zeros(np.squeeze(results[4]).shape, dtype=np.int32)]
        scale_factor = 2
        search_radius = 600
        left_pyramid_scaled = [scipy.ndimage.zoom(left_pyramid[0], [1./scale_factor, 1./scale_factor, 1], order=1)]
        right_pyramid_scaled = [scipy.ndimage.zoom(right_pyramid[0], [1./scale_factor, 1./scale_factor, 1], order=1)]
        occluded_pyramid_scaled = [scipy.ndimage.zoom(occluded_pyramid[0], [1./scale_factor, 1./scale_factor], order=1).astype(np.int32)]
        predicted_flow_small, feature_errors, flow_arrays_unscaled = triplet_flow_loss.run_slow_flow_calculator_process.get_flow_parallel_pyramid(left_pyramid_scaled, right_pyramid_scaled,
                                           occluded_pyramid_scaled, neighborhood_len_import=search_radius/scale_factor, interpolate_after=False)
        predicted_flow = scipy.ndimage.zoom(predicted_flow_small, [scale_factor, scale_factor, 1], order=1) * scale_factor
        feature_errors = scipy.ndimage.zoom(feature_errors, [scale_factor, scale_factor], order=1)
        flow_arrays = list()
        for flow_arr in flow_arrays_unscaled:
            flow_arrays.append(scipy.ndimage.zoom(flow_arr, [scale_factor, scale_factor, 1], order=1) * scale_factor)

        # # Calculate flow from right to left image then map it to the right image
        # occluded_pyramid_scaled_r = [scipy.ndimage.zoom(1 - right_labels_blob[0, :, :, 1], [1. / scale_factor, 1. / scale_factor], order=1).astype(np.int32)]
        # r_to_l_predicted_flow, _, _ = triplet_flow_loss.run_slow_flow_calculator_process.get_flow_parallel_pyramid(right_pyramid_scaled, left_pyramid_scaled,
        #                                              occluded_pyramid_scaled_r, neighborhood_len_import=search_radius/scale_factor, interpolate_after=False)
        # r_to_l_predicted_flow_l_view = np.zeros(r_to_l_predicted_flow.shape, dtype=np.float32)
        # for y in range(r_to_l_predicted_flow.shape[0]):
        #     for x in range(r_to_l_predicted_flow.shape[1]):
        #         orrigional_point_y = int(round(r_to_l_predicted_flow[int(y), int(x), 1])) + y
        #         orrigional_point_x = int(round(r_to_l_predicted_flow[int(y), int(x), 0])) + x
        #         if orrigional_point_x == x and orrigional_point_y == y:
        #             continue
        #         if 0 <= orrigional_point_y < r_to_l_predicted_flow.shape[0] and 0 <= orrigional_point_x < r_to_l_predicted_flow.shape[1]:
        #             r_to_l_predicted_flow_l_view[orrigional_point_y, orrigional_point_x] = [x - orrigional_point_x, y - orrigional_point_y]

        # # Pyramidal flow calculation
        # left_pyramid = (np.squeeze(results[7]), np.squeeze(results[6]), np.squeeze(results[5]), np.squeeze(results[0]))
        # right_pyramid = (np.squeeze(results[10]), np.squeeze(results[9]), np.squeeze(results[8]), np.squeeze(results[1]))
        # occluded_pyramid = (np.squeeze(results[13]), np.squeeze(results[12]), np.squeeze(results[11]), np.squeeze(results[4]))
        # predicted_flow, feature_errors, flow_arrays = triplet_flow_loss.run_slow_flow_calculator_process.get_flow_parallel_pyramid(left_pyramid, right_pyramid,
        #                                    occluded_pyramid, neighborhood_len_import=400, interpolate_after=True)
        # # predicted_flow = interpolate_flow(np.squeeze(results[0]), np.squeeze(results[1]), predicted_flow)

        # calculate EPE, not counting occluded regions
        predicted_flow_cropped = predicted_flow[:gt_flow.shape[0], :gt_flow.shape[1]]
        gt_flow_cropped = gt_flow[:predicted_flow.shape[0], :predicted_flow.shape[1]]
        mask_cropped = np.squeeze(results[4])[:predicted_flow.shape[0], :predicted_flow.shape[1]]
        predicted_flow_masked = np.where(np.dstack([mask_cropped, mask_cropped]) == 0, predicted_flow_cropped, np.nan)
        l2_dist_arr = np.sqrt(np.sum(np.square(predicted_flow_masked - gt_flow_cropped), axis=2))
        average_EPE = np.nanmean(l2_dist_arr)

        if calculate_EPE_all_data:
            path_segments = str(images['image']).split("/")
            print ("%3i / %i EPE is %7.4f for " % (image_index_pos + 1, n_images, average_EPE)) + path_segments[-3] + "/" + path_segments[-2] + "/" + path_segments[-1]
            print "\tcalculated triplet loss is %7.4f" % float(results[3][0])
            EPE_list.append(average_EPE)
            try:
                class_epe_set[blobs['roidb'][0]['video_id']][0] += average_EPE
                class_epe_set[blobs['roidb'][0]['video_id']][1] += 1
            except:
                class_epe_set[blobs['roidb'][0]['video_id']] = list([average_EPE, 1])
            print blobs['roidb'][0]['video_id'], class_epe_set[blobs['roidb'][0]['video_id']]
        else:
            global iiiiii, x_plots, y_plots, axes_left_list, axes_right_list, fig  # using a global variable probably isn't the best here, but it works.
            fig = plt.figure(figsize=(12.0, 9.0))
            iiiiii = 1
            axes_left_list = list()
            axes_right_list = list()

            if show_arrows:
                x_plots = 1
                y_plots = 2

                # crop the images so they aren't quite as tiny
                left_im = fix_rgb_image(left_blob[0])
                right_im = fix_rgb_image(right_blob[0])

                # left_coords = np.argwhere(np.sum(left_labels_blob[0], axis=2) > 0)
                # right_coords = np.argwhere(np.sum(right_labels_blob[0], axis=2) > 0)
                #
                # x0, y0 = np.min([right_coords.min(axis=0), left_coords.min(axis=0)], axis=0)
                # x1, y1 = np.max([right_coords.max(axis=0), left_coords.max(axis=0)], axis=0) + 1
                # x0 = np.max([x0 - 20, 0])
                # y0 = np.max([y0 - 20, 0])
                # x1 = np.min([x1 + 20, left_im.shape[0]])
                # y1 = np.min([y1 + 20, left_im.shape[1]])
                #
                # left_im = left_im[x0:x1, y0:y1]
                # right_im = right_im[x0:x1, y0:y1]
                #
                # left_labels = left_labels_blob[0][x0:x1, y0:y1, 1:]
                # occluded = occluded_blob[0][x0:x1, y0:y1, 0]
                # gt_flow_ = gt_flow[x0:x1, y0:y1, :]
                # predicted_flow_ = predicted_flow[x0:x1, y0:y1, :]

                left_im = left_im
                right_im = right_im

                left_labels = left_labels_blob[0]
                occluded = occluded_blob[0][:, :, 0]
                gt_flow_ = gt_flow
                predicted_flow_ = predicted_flow


                lr_arr = np.concatenate([left_im, right_im], axis=1)
                lr_img = lr_arr
                ax1 = fig.add_subplot(y_plots, x_plots, iiiiii)
                ax1.imshow(lr_img)
                ax1.set_title("left image, right image")
                iiiiii += 1

                flow_arr = np.concatenate([gt_flow_, predicted_flow_], axis=1)
                flow_img = sintel_utils.sintel_compute_color(flow_arr)
                ax2 = fig.add_subplot(y_plots, x_plots, iiiiii)
                ax2.imshow(flow_img, )
                ax2.set_title("gt flow, predicted flow")
                iiiiii += 1

                label_arr = np.sum(left_labels, axis=2)  # gets rid of background
                # occluded = occluded[0][:, :, 0]  # to get rid of len one dimension on the end
                n_arrows = 8
                spacing = int(np.sqrt(np.count_nonzero((1 - occluded) * label_arr) / n_arrows))
                for x in range(0, gt_flow_.shape[1], spacing):
                    for y in range(int((x * 0.29) % (spacing / 2)), gt_flow_.shape[0], spacing):
                        x_ = max(0, min(x + random.randint(-1 * spacing / 4, spacing / 4), gt_flow_.shape[1] - 1))
                        y_ = max(0, min(y + random.randint(-1 * spacing / 4, spacing / 4), gt_flow_.shape[0] - 1))
                        if occluded[y_, x_] == 0 and label_arr[y_, x_] != 0:
                            r_offset = predicted_flow_[y_, x_]
                            r_offset[0] += int(gt_flow_.shape[1]) + x_
                            r_offset[1] += y_

                            color = np.random.rand(3, )
                            ax1.plot([x_, r_offset[0]], [y_, r_offset[1]], 'k-', lw=1.0, color=color)
                            ax1.plot([x_, r_offset[0]], [y_, r_offset[1]], marker='o', color=color, markersize=3)

            else:
                x_plots = 3
                y_plots = 3

                # show left
                im_left = fix_rgb_image(left_blob[0])
                ax1 = fig.add_subplot(y_plots, x_plots, iiiiii)
                ax1.imshow(im_left)
                ax1.set_title("left image")
                iiiiii += 1
                axes_left_list.append(ax1)

                # show right
                im_right = fix_rgb_image(right_blob[0])
                ax2 = fig.add_subplot(y_plots, x_plots, iiiiii)
                ax2.imshow(im_right)
                ax2.set_title("right image (red dot is predicted flow, green is ground truth)")
                iiiiii += 1
                axes_right_list.append(ax2)

                # show occluded
                ax2 = fig.add_subplot(y_plots, x_plots, iiiiii)
                ax2.imshow(((np.squeeze(occluded_blob[0]) * -1 + 1)[:, :, np.newaxis] * im_left).astype(np.uint8))
                ax2.set_title("occluded")
                iiiiii += 1
                axes_left_list.append(ax2)

                # create flow images, but don't display them yet
                gt_flow_raw_color = sintel_utils.raw_color_from_flow(gt_flow)
                gt_flow_plot_position = iiiiii
                iiiiii += 1

                predicted_flow_ = predicted_flow[x0:x1, y0:y1, :]
                computed_flows_plot_positions = list()
                computed_flows_color_square = list()
                computed_flows_raw_color = list()

                combined_im = sintel_utils.sintel_compute_color(np.concatenate([gt_flow, predicted_flow], axis=1))

                gt_flow_color_square = combined_im[:, :640, :]
                pred_flow_im = combined_im[:, 640:, :]

                computed_flows_color_square.append(pred_flow_im)
                computed_flows_raw_color.append(sintel_utils.raw_color_from_flow(predicted_flow))
                computed_flows_plot_positions.append(iiiiii)
                iiiiii += 1

                # display_img(sintel_utils.sintel_compute_color(predicted_flow_l), "lef to right flow (left)", right=True)
                # display_img(sintel_utils.sintel_compute_color(r_to_l_predicted_flow), "right to left flow (right)", right=True)
                # display_img(sintel_utils.sintel_compute_color(r_to_l_predicted_flow_l_view), "right to left flow (left)")
                # display_img(np.sqrt(np.sum(np.power(r_to_l_predicted_flow_l_view - predicted_flow, 2), axis=2)), "right to left flow similarity")

                for image_index_pos in range(len(flow_arrays) - 1):
                    computed_flows_color_square.append(sintel_utils.sintel_compute_color(flow_arrays[image_index_pos]))
                    computed_flows_raw_color.append(sintel_utils.raw_color_from_flow(flow_arrays[image_index_pos]))
                    computed_flows_plot_positions.append(iiiiii)
                    iiiiii += 1

                gt_flow_ax = fig.add_subplot(y_plots, x_plots, gt_flow_plot_position)

                # Using a closure so that we don't have to make gt_flow_ax a global variable. It can't be passed as an
                # argument because plot_flow_images will be called in a matplotlib event handler
                def get_plot_flow_images(gt_flow_ax):
                    def plot_flow_images(color_square_not_raw):
                        if color_square_not_raw:
                            gt_flow_ax.imshow(gt_flow_color_square)
                        else:
                            gt_flow_ax.imshow(gt_flow_raw_color)
                        gt_flow_ax.set_title("gt flow")
                        axes_left_list.append(gt_flow_ax)

                        for ii in range(len(flow_arrays)):
                            ax7 = fig.add_subplot(y_plots, x_plots, computed_flows_plot_positions[ii])
                            if color_square_not_raw:
                                ax7.imshow(computed_flows_color_square[ii])
                            else:
                                ax7.imshow(computed_flows_raw_color[ii])
                            ax7.set_title("raw predicted flow at scale " + str(ii))
                            axes_left_list.append(ax7)
                    return plot_flow_images

                plot_flow_images = get_plot_flow_images(gt_flow_ax)

                plot_flow_images(True)

                # # display depth
                # ax2 = fig.add_subplot(y_plots, x_plots, iiiiii)
                # ax2.imshow(np.squeeze(depth_blob[0]))
                # ax2.set_title("depth")
                # iiiiii += 1
                # axes_left_list.append(ax2)

                # left_labels = (left_labels_blob[0] * np.arange(1, left_labels_blob[0].shape[2] + 1)).sum(axis=2)
                # right_labels = (right_labels_blob[0] * np.arange(1, right_labels_blob[0].shape[2] + 1)).sum(axis=2)
                # display_img(np.squeeze(left_labels), "left labels")
                # display_img(np.squeeze(right_labels), "right labels", right=True)

                random_matrix = np.random.rand(features_l.shape[2], 3)
                feature_ax_list = list()
                feature_ax_list.append([np.squeeze(features_l), "left features", None, False])
                feature_ax_list.append([np.squeeze(features_r), "right features", None, True])
                # feature_ax_list.append([np.squeeze(results[7]), "left features 8x", None, False])
                # feature_ax_list.append([np.squeeze(results[10]), "right features 8x", None, True])
                # feature_ax_list.append([np.squeeze(results[6]), "left features 4x", None, False])
                # feature_ax_list.append([np.squeeze(results[9]), "right features 4x", None, True])
                # feature_ax_list.append([np.squeeze(results[5]), "left features 2x", None, False])
                # feature_ax_list.append([np.squeeze(results[8]), "right features 2x", None, True])
                # feature_ax_list.append([np.squeeze(results[0]), "left features 1x", None, False])
                # feature_ax_list.append([np.squeeze(results[1]), "right features 1x", None, True])
                for ii in range(len(feature_ax_list)):
                    iii = feature_ax_list[ii]
                    ax = display_img(sintel_utils.colorize_features(np.matmul(iii[0], random_matrix[:iii[0].shape[2], :])), iii[1], right=iii[3])
                    feature_ax_list[ii] = [iii[0], iii[1], ax, iii[3]]


                # neighboring feature differences
                feature_similarity = np.zeros(features_l.shape[:2], dtype=np.float32)
                radius = 4  # actually half the side length of the square used for sampling
                stride = 2
                for a in range(radius, feature_similarity.shape[0] - radius, stride):
                    for b in range(radius, feature_similarity.shape[1] - radius, stride):
                        feature_similarity[a : a + stride, b : b + stride] = np.average(np.abs(features_l[a - radius : a + radius, b - radius : b + radius] - features_l[a, b]))
                display_img(feature_similarity, "neighboring feature difference (l1 distance)")


                # Store point_list in a closure because onclick will be called by a matplotlib event handler and we
                # can't pass it any arguments
                def get_onclick():
                    # So variables can persist between calls
                    point_list = list()
                    x_points = list()
                    y_points = list()
                    x_points_right = list()
                    y_points_right = list()
                    def onclick(event):
                        print(
                        'button', event.button, 'x=', event.x, 'y=', event.y, 'xdata=', event.xdata, 'ydata=', event.ydata)
                        for ax in axes_left_list:
                            data_transformer = ax.transData.inverted()
                            x_left, y_left = data_transformer.transform([event.x, event.y])
                            if -1 <= x_left <= im_left.shape[1] + 3 and -1 <= y_left <= im_left.shape[0] + 3:
                                print("\t x transformed:", x_left, "y transformed:", y_left)
                                if event.xdata is not None and event.ydata is not None:
                                    if occluded_blob[0][int(event.ydata), int(event.xdata)] != 1 or True:
                                        x_point = event.xdata / (np.max(ax.get_xlim()) + 0.5) * 640
                                        y_point = event.ydata / (np.max(ax.get_ylim()) + 0.5) * 480
                                        x_points_right.append(x_point + int(gt_flow[int(y_point), int(x_point), 0]))
                                        y_points_right.append(y_point + int(gt_flow[int(y_point), int(x_point), 1]))
                                        x_points.append(x_point)
                                        y_points.append(y_point)
                                        color = random.random()

                                        try:
                                            l_feature = results[0][0, int(y_point), int(x_point)]
                                            r_feature_gt = results[1][0, int(y_points_right[-1]), int(x_points_right[-1])]
                                            r_feature_pred = results[1][0, int(y_point) + int(predicted_flow[int(y_point), int(x_point), 1]),
                                                                        int(x_point) + int(predicted_flow[int(y_point), int(x_point), 0])]
                                            print "dist between gt l and r features is", np.sqrt(np.sum(np.power(l_feature - r_feature_gt, 2)))
                                            print "dist between predicted l and r features is", np.sqrt(np.sum(np.power(l_feature - r_feature_pred, 2)))
                                        except:
                                            pass

                                        for sub_ax in axes_left_list:
                                            sub_ax.scatter([x_point / 640 * (np.max(sub_ax.get_xlim()) + 0.5)],
                                                           [y_point / 480 * (np.max(sub_ax.get_ylim()) + 0.5)], c=[color], s=7, marker='1')

                                        for ii in point_list:
                                            ii.remove()
                                        while len(point_list) > 0:
                                            point_list.pop()
                                        for sub_ax in axes_right_list:
                                            point_list.append(sub_ax.scatter([(x_point + int(predicted_flow[int(y_point), int(x_point), 0])) / 640 * (np.max(sub_ax.get_xlim()) + 0.5)],
                                                           [(y_point + int(predicted_flow[int(y_point), int(x_point), 1])) / 640 * (np.max(sub_ax.get_xlim()) + 0.5)], c='RED', edgecolors='WHITE', s=8, marker='o'))
                                            point_list.append(sub_ax.scatter([(x_point + int(gt_flow[int(y_point), int(x_point), 0])) / 640 * (np.max(sub_ax.get_xlim()) + 0.5)],
                                                           [(y_point + int(gt_flow[int(y_point), int(x_point), 1])) / 640 * (np.max(sub_ax.get_xlim()) + 0.5)], c='GREEN', s=5, marker='1'))
                                    else:
                                        print "Point occluded, not drawing"
                                    fig.canvas.draw()
                                break
                    return onclick

                fig.canvas.mpl_connect('button_press_event', get_onclick())

                def get_handle_key_press(feature_ax_list):
                    # So triplets can persist between calls
                    triplet_container = list()
                    triplet_container.append(None)

                    def handle_key_press(event):
                        if event.key == 'c':
                            plot_flow_images(True)
                        elif event.key == 'r':
                            plot_flow_images(False)

                        elif event.key == 'e':
                            random_matrix = np.random.rand(features_l.shape[2], 3)
                            for ii in feature_ax_list:
                                display_img(sintel_utils.colorize_features(np.matmul(ii[0], random_matrix[:ii[0].shape[2], :])), ii[1], ax=ii[2], right=ii[3])

                        elif event.key == "y":
                            string_arr = pyperclip.paste()
                            tup = literal_eval(string_arr)
                            triplets = np.array(tup)
                            triplets = triplets.transpose()
                            triplet_container[0] = triplets

                        elif event.key == "t":
                            triplets = triplet_container[0]
                            try:
                                x_a = triplets[0, 0] % 640
                                y_a = triplets[0, 0] / 640
                                x_p = triplets[1, 0] % 640
                                y_p = triplets[1, 0] / 640
                                x_n = triplets[2, 0] % 640
                                y_n = triplets[2, 0] / 640
                                print "x_a", x_a, "y_a", y_a, "x_p", x_p, "y_p", y_p, "x_n", x_n, "y_n", y_n
                                for sub_ax in axes_left_list:
                                    sub_ax.scatter([x_a], [y_a], c="BLUE", s=6, marker='p')
                                    sub_ax.scatter([x_n], [y_n], c="RED", s=4, marker='p')
                                for sub_ax in axes_right_list:
                                    sub_ax.scatter([x_p], [y_p], c="GREEN", s=4, marker='p')
                                fig.canvas.draw()

                                triplets = triplets[:, 1:]
                                print triplets.shape[1], "points left"
                            except:
                                print "data must be loaded with l before it can be plotted"
                        elif event.key == "b":
                            global image_index_pos
                            image_index_pos =- 1
                            print "stepped backwards in image set, image_index_pos now equals", image_index_pos

                        else:
                            print "key not tied to any action"
                        # only redraw if something changed
                        fig.canvas.draw()

                    return handle_key_press

                fig.canvas.mpl_connect('key_press_event', get_handle_key_press(feature_ax_list))

            fig.suptitle("Left Image: " + str(blobs['roidb'][0]['image'].split("/")[-2:]) + "  \nright image: " +
                         str(blobs['roidb'][0]['image_right'].split("/")[-2:]) + " triplet_loss:" + str(
                        results[3][0]) + " EPE:" + str(average_EPE))

            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.1, hspace=0.2)
            if save_image:
                plt.savefig("plot_" + str(image_index_pos) + ".png")
            else:
                plt.show()
            plt.close('all')

    if calculate_EPE_all_data:
        average = np.mean(EPE_list)
        print "# average EPE is " + str(average) + " for entire " + str(imdb._name) + " dataset with network " + \
            str(weights_filename)
        for video_id in class_epe_set:
            print video_id, "has average EPE", class_epe_set[video_id][0] / class_epe_set[video_id][1]


def display_img(img, title, right=False, ax=None):
    global iiiiii, x_plots, y_plots, axes_left_list, axes_right_list, fig
    if ax is None:
        ax = fig.add_subplot(y_plots, x_plots, iiiiii)
        iiiiii += 1
        if right:
            axes_right_list.append(ax)
        else:
            axes_left_list.append(ax)
    ax.imshow(img)
    ax.set_title(title)
    return ax


# vgg normalizes images before inputting them, this denormalizes the images back to RGB
def fix_rgb_image(image_in):
    image = image_in.copy() + cfg.PIXEL_MEANS
    image = image[:, :, (2, 1, 0)]
    image = image.astype(np.uint8)
    return image


# def calculate_flow_single_frame(sess, net, im_left, im_right):
#     # compute image blob
#     left_blob, right_blob, im_scales = _get_flow_image_blob(im_left, im_right, 0)
#
#     feed_dict = {net.data_left: left_blob, net.data_right: right_blob,
#                  net.gt_flow: np.zeros([left_blob.shape[0], left_blob.shape[1], left_blob.shape[2], 2],
#                                        dtype=np.float32), net.keep_prob: 1.0}
#
#     sess.run(net.enqueue_op, feed_dict=feed_dict)
#     output_flow = sess.run([net.get_output('predicted_flow')])
#     return output_flow
#
#
# def siphon_flow_single_frame(sess, net, im_left, im_right):
#     # compute image blob
#     left_blob, right_blob, im_scales = _get_flow_image_blob(im_left, im_right, 0)
#
#     training_data_queue = list()
#     queue_start_size = sess.run(net.queue_size_op)
#     while sess.run(net.queue_size_op) != 0:
#         training_data_queue.append(sess.run({'left':net.get_output('data_left'), 'right':net.get_output('data_right'),
#                                              'flow':net.get_output('gt_flow'), 'keep_prob':net.keep_prob_queue}))
#
#     feed_dict = {net.data_left: left_blob, net.data_right: right_blob,
#                  net.gt_flow: np.zeros([left_blob.shape[0], left_blob.shape[1], left_blob.shape[2], 2],
#                                        dtype=np.float32), net.keep_prob: 1.0}
#
#     sess.run(net.enqueue_op, feed_dict=feed_dict)
#     output = sess.run({'flow':net.get_output('predicted_flow'), 'left':net.get_output('data_left_tap'),
#                             'right':net.get_output('data_left_tap')})
#
#     for i in training_data_queue:
#         feed_dict = {net.data_left: i['left'], net.data_right: i['right'], net.gt_flow: i['flow'], net.keep_prob: i['keep_prob']}
#         sess.run(net.enqueue_op, feed_dict=feed_dict)
#
#     # assert sess.run(net.queue_size_op) == queue_start_size, "data queue size changed"
#     return output


def siphon_outputs_single_frame(sess, net, data_feed_dict, outputs):
    # compute image blob

    training_data_queue = list()
    queue_start_size = sess.run(net.queue_size_op)
    while sess.run(net.queue_size_op) != 0:
        training_data_queue.append(sess.run({'left':net.get_output('data_left'), 'right':net.get_output('data_right'),
                                             'flow':net.get_output('gt_flow'), 'keep_prob':net.keep_prob_queue}))


    sess.run(net.enqueue_op, feed_dict=data_feed_dict)
    output = sess.run(outputs)

    for i in training_data_queue:
        feed_dict = {net.data_left: i['left'], net.data_right: i['right'], net.gt_flow: i['flow'], net.keep_prob: i['keep_prob']}
        sess.run(net.enqueue_op, feed_dict=feed_dict)

    # assert sess.run(net.queue_size_op) == queue_start_size, "data queue size changed"
    return output


# def _get_flow_image_blob(im_left, im_right, scale_ind):
#     """Converts an image into a network input.
#
#     Arguments:
#         im (ndarray): a color image in BGR order
#
#     Returns:
#         blob (ndarray): a data blob holding an image pyramid
#         im_scale_factors (list): list of image scales (relative to im) used
#             in the image pyramid
#     """
#     num_images = 1
#     processed_left = []
#     processed_right = []
#     processed_flow = []
#     im_scales = []
#
#     # left image
#     im_left = pad_im(cv2.imread(im_left, cv2.IMREAD_UNCHANGED), 16)
#     if im_left.shape[2] == 4:
#         im = np.copy(im_left[:, :, :3])
#         alpha = im_left[:, :, 3]
#         I = np.where(alpha == 0)
#         im[I[0], I[1], :] = 0
#         im_lef = im
#
#     im_right = pad_im(cv2.imread(im_right, cv2.IMREAD_UNCHANGED), 16)
#     if im_left.shape[2] == 4:
#         im = np.copy(im_left[:, :, :3])
#         alpha = im_right[:, :, 3]
#         I = np.where(alpha == 0)
#         im[I[0], I[1], :] = 0
#         im_right = im
#
#
#     # TODO: is this important?
#     im_scale = cfg.TEST.SCALES_BASE[scale_ind]
#     im_scales.append(im_scale)
#
#     im_left_orig = im_left.astype(np.float32, copy=True)
#     im_left_orig -= cfg.PIXEL_MEANS
#     im_left_processed = cv2.resize(im_left_orig, None, None, fx=im_scale, fy=im_scale,
#                                    interpolation=cv2.INTER_LINEAR)
#     processed_left.append(im_left_processed)
#
#     im_right_orig = im_right.astype(np.float32, copy=True)
#     im_right_orig -= cfg.PIXEL_MEANS
#     im_right_processed = cv2.resize(im_right_orig, None, None, fx=im_scale, fy=im_scale,
#                                     interpolation=cv2.INTER_LINEAR)
#     processed_right.append(im_right_processed)
#
#
#     # Create a blob to hold the input images
#     image_left_blob = im_list_to_blob(processed_left, 3)
#     image_right_blob = im_list_to_blob(processed_right, 3)
#     blob_rescale = []
#
#     return image_left_blob, image_right_blob, im_scales