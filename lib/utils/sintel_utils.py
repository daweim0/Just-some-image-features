import numpy as np
import struct
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imread
import time
import random

MAXCOLS = 60
ncols = 0
colorwheel = None


def read_flow_file(base_path, scene, n):
    return read_flow_file_with_path(base_path + "/flow/" + scene + "/frame_%0.4i" % n + ".flo")


def read_flow_file_with_path(path):
    with open(path, "rb") as flow_file:
        raw_data = flow_file.read()
        assert struct.unpack_from("f", raw_data, 0)[0] == 202021.25  # check to make sure the file is being read correctly
        width, height = struct.unpack_from("ii", raw_data, 4)
        data_raw = np.frombuffer(raw_data, dtype=np.float32, offset=12)
        data = data_raw.reshape([height, width, 2], order='C').transpose([1, 0, 2])
        return data


def setcols(r, g, b, k):
    colorwheel[k][0] = r
    colorwheel[k][1] = g
    colorwheel[k][2] = b


def makecolorwheel():
    # relative lengths of color transitions:
    # these are chosen based on perceptual similarity
    # (e.g. one can distinguish more shades between red and yellow
    #  than between yellow and green)
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    global ncols, colorwheel
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3], dtype=np.float32)
    print "ncols = %d\n" % ncols
    if (ncols > MAXCOLS):
        raise EnvironmentError("something went wrong?")
    k = 0
    for i in range(RY):
        setcols(1,	   1.0*float(i)/RY,	 0,	       k)
        k += 1
    for i in range(YG):
        setcols(1.0-float(i)/YG, 1,		 0,	       k)
        k += 1
    for i in range(GC):
        setcols(0,		   1,		 float(i)/GC,     k)
        k += 1
    for i in range(CB):
        setcols(0,		   1-float(i)/CB, 1,	       k)
        k += 1
    for i in range(BM):
        setcols(float(i)/BM,	   0,		 1,	       k)
        k += 1
    for i in range(MR):
        setcols(1,	   0,		 1-float(i)/MR, k)
        k += 1
makecolorwheel()


def sintel_compute_color(data_interlaced):
    # type: (np.ndarray) -> np.ndarray
    data_u_in, data_v_in = np.split(data_interlaced, 2, axis=2)
    data_u_in = np.squeeze(data_u_in)
    data_v_in = np.squeeze(data_v_in)
    # pre-normalize (for some reason?)
    max_rad = np.max(np.sqrt(np.power(data_u_in, 2) + np.power(data_v_in, 2)))
    fx = data_u_in / max_rad
    fy = data_v_in / max_rad

    # now do the stuff done in computeColor()
    rad = np.sqrt(np.power(fx, 2) + np.power(fy, 2))
    a = np.arctan2(-fy, -fx) / np.pi
    fk = (a + 1.0) / 2.0 * (ncols-1)
    k0 = fk.astype(np.int32)
    k1 = ((k0 + 1) % ncols).astype(np.int32)
    f = fk - k0
    h, w = k0.shape
    col0 = colorwheel[k0.reshape(-1)].reshape([h, w, 3])
    col1 = colorwheel[k1.reshape(-1)].reshape([h, w, 3])
    col = (1 - f[..., np.newaxis]) * col0 + f[..., np.newaxis] * col1
    # col = col0

    col = 1 - rad[..., np.newaxis] * (1 - col)  # increase saturation with radius
    return col


if __name__ == "__main__":
    frame_number = 4
    training_set = "ambush_4"
    flow_data = read_flow_file("data/sintel/training", training_set, frame_number).transpose([1, 0, 2])
    rgb_new = sintel_compute_color(flow_data)
    # plt.imshow(np.dstack([np.transpose(rgb_new, [1, 0, 2]), np.ones([rgb_new.shape[1], rgb_new.shape[0], 1])]))
    # image_ref = mpimg.imread("data/sintel/training/flow_viz/" + training_set + ("/frame_%0.4i.png" % frame_number))
    image_left = imread("data/sintel/training/albedo/" + training_set + ("/frame_%0.4i.png" % frame_number))
    image_right = imread("data/sintel/training/albedo/" + training_set + ("/frame_%0.4i.png" % (frame_number + 1)))
    occlusions = imread("data/sintel/training/occlusions/" + training_set + ("/frame_%0.4i.png" % (frame_number)))
    image_left = image_left.astype(np.float32) / 255.0
    image_right = image_right.astype(np.float32) / 255.0
    occlusions = occlusions.astype(np.float32) / 255.0
    valid_mask = (occlusions - 1) * -1

    multipliers = [1.0]
    # warp left image
    # warped = np.zeros(list([len(multipliers)]) + list(image_left.shape))
    # # warped = np.copy(image_left)
    # # warped = np.copy(image_right)
    # for multiplier_ind in range(len(multipliers)):
    #     multiplier = multipliers[multiplier_ind]
    #     for i in range(image_left.shape[0]):
    #         for j in range(image_left.shape[1]):
    #             i_new = i + int(flow_data[i, j, 0] * multiplier)
    #             j_new = j + int(flow_data[i, j, 1] * multiplier)
    #             if 0 <= i_new < warped.shape[1] and 0 <= j_new < warped.shape[2]:
    #                 warped[multiplier_ind, i_new, j_new] = image_left[i, j]


    # print "average error:", np.average(rgb_new - image_ref.transpose([1,0,2]))

    image_both = np.concatenate([image_left, image_right], axis=0)

    i = 1
    plot_x = 1
    plot_y = 2

    fig = plt.figure()
    plt.ion()

    # ax1 = fig.subplot(320 + i)
    # ax1.imshow(rgb_new)
    # ax1.set_title("calculated image")
    #
    # ax2 = plt.subplot(plot_y, plot_x, i)
    # ax2.imshow(image_left)
    # ax2.set_title("left image")
    # i += 1
    #
    # ax3 = plt.subplot(plot_y, plot_x, i)
    # ax3.imshow(image_right)
    # ax3.set_title("right image")
    # i += 1

    ax_both = plt.subplot(plot_y, plot_x, i)
    ax_both.imshow(image_both, interpolation='nearest')
    ax_both.set_title("left on top, right on bottom")
    i += 1
    #
    # ax6 = fig.subplot(plot_y, plot_x, i)
    # ax6.imshow(valid_mask)
    # ax6.set_title("valid mask")
    # i += 1
    #
    # ax6 = fig.subplot(plot_y, plot_x, i)
    # ax6.imshow(occlusions[:,:,np.newaxis] * image_right)
    # ax6.set_title("occlusions * right_image")
    # i += 1

    # for index in range(len(multipliers)):
    #     ax4 = fig.subplot(plot_y, plot_x, i)
    #     ax4.imshow(warped[index][:,200:900,:])
    #     ax4.set_title("warped multiplier " + str(multipliers[index]))
    #     i += 1

        # ax5 = fig.subplot(plot_y, plot_x, i)
        # ax5.imshow(np.abs((image_right - warped[index]) * valid_mask[:,:,np.newaxis]))
        # ax5.set_title("difference multiplier " + str(multipliers[index]))
        # i += 1


    # rgb_old = old_compute_color(np.dstack([data_u, data_v]))
    # ax4 = fig.subplot(224)
    # ax4.imshow(rgb_new.transpose([1, 0, 2]))
    # ax4.set_title("non-vectorized image")

    # fig.gray()
    # plt.tight_layout()
    plt.gca().set_position([0.05, 0.05, 0.95, 0.95])


    x_points = list()
    y_points = list()
    x_points_right = list()
    y_points_right = list()
    colors = list()


    def onclick(event):
        print('button', event.button, 'x=', event.x, 'y=', event.y, 'xdata=', event.xdata, 'ydata=', event.ydata)
        data_transformer = ax_both.transData.inverted()
        # x_left, y_left = data_transformer.transform([event.x, event.y])
        # print("\t x transformed:", x_left, "y transformed:", y_left)
        # if 0 <= x_left < image_left.shape[1] and 0 <= y_left < image_left.shape[0]:
        if event.xdata is not None and event.ydata is not None:
            x_points_right.append(event.xdata + int(flow_data[int(event.ydata), int(event.xdata), 0]))
            y_points_right.append(event.ydata + int(flow_data[int(event.ydata), int(event.xdata), 1]) + image_left.shape[0])
            x_points.append(event.xdata)
            y_points.append(event.ydata)
            colors.append(random.random())

            ax_both.scatter(x_points, y_points, c=colors, s=2, marker='p')
            ax_both.scatter(x_points_right, y_points_right, c=colors, s=2, marker='p')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)


    # plt.show()
    while True:
        plt.pause(0.001)
    pass
