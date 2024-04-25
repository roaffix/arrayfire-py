#!/usr/bin/env python

#######################################################
# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from time import time
import arrayfire as af
import os
import sys


def draw_corners(img, x, y, draw_len):
    # Draw vertical line of (draw_len * 2 + 1) pixels centered on  the corner
    # Set only channel 1 to 1 (green lines)
    xmin = max(0, x - draw_len)
    xmax = min(img.shape[1], x + draw_len)

    img[y, xmin : xmax, 0] = 0.0
    img[y, xmin : xmax, 1] = 1.0
    img[y, xmin : xmax, 2] = 0.0

    # Draw vertical line of (draw_len * 2 + 1) pixels centered on  the corner
    # Set only the first channel to 1 (green lines)
    ymin = max(0, y - draw_len)
    ymax = min(img.shape[0], y + draw_len)

    img[ymin : ymax, x, 0] = 0.0
    img[ymin : ymax, x, 1] = 1.0
    img[ymin : ymax, x, 2] = 0.0
    return img

def harris_demo(console):

    root_path = os.path.dirname(os.path.abspath(__file__))
    file_path = root_path
    if console:
        file_path += "/../../assets/examples/images/square.png"
    else:
        file_path += "/../../assets/examples/images/man.jpg"
    img_color = af.load_image(file_path, is_color=True);

    img = af.color_space(img_color, af.CSpace.GRAY, af.CSpace.RGB)
    img_color /= 255.0

    ix, iy = af.gradient(img)
    ixx = ix * ix
    ixy = ix * iy
    iyy = iy * iy

    # Compute a Gaussian kernel with standard deviation of 1.0 and length of 5 pixels
    # These values can be changed to use a smaller or larger window
    gauss_filt = af.gaussian_kernel(5, 5, rows_sigma=1.0, columns_sigma=1.0)

    # Filter second order derivatives
    ixx = af.convolve2(ixx, gauss_filt)
    ixy = af.convolve2(ixy, gauss_filt)
    iyy = af.convolve2(iyy, gauss_filt)

    # Calculate trace
    itr = ixx + iyy

    # Calculate determinant
    idet = ixx * iyy - ixy * ixy

    # Calculate Harris response
    response = idet - 0.04 * (itr * itr)

    # Get maximum response for each 3x3 neighborhood
    msk = af.constant(1, (3, 3))
    max_resp = af.dilate(response, mask=msk)

    # Discard responses that are not greater than threshold
    corners = response > float(1e5)
    corners = corners * response

    # Discard responses that are not equal to maximum neighborhood response,
    # scale them to original value
    corners = (corners == max_resp) * corners

    # Copy device array to python list on host
    corners_list = corners.copy()
    # import pdb; pdb.set_trace()

    draw_len = 3
    good_corners = 0
    for x in range(img_color.shape[1] - 1):
        for y in range(img_color.shape[0] - 1):
            # print(f"x:{x}, y:{y}")
            # # print(corners_list[x, y])
            if (0 <= x < corners_list.shape[0]) and (0 <= y < corners_list.shape[1]):
                if corners_list[x, y] > 1e5:
                    img_color = draw_corners(img_color, x, y, draw_len)
                    good_corners += 1
            else:
                continue


    print("Corners found: {}".format(good_corners))
    if not console:
        # Previews color image with green crosshairs
        file_path = os.path.join(os.getcwd(), 'harris_image.png')
        af.save_image(img_color, file_path)
    else:
        idx = af.where(corners)

        corners_x = idx / float(corners.dims()[0])
        corners_y = idx % float(corners.dims()[0])

        print(corners_x)
        print(corners_y)


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        af.set_device(int(sys.argv[1]))
    console = (sys.argv[2] == '-') if len(sys.argv) > 2 else False

    af.info()
    print("** ArrayFire Harris Corner Detector Demo **\n")

    harris_demo(console)