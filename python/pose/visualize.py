import cv2
import math
import numpy as np

import os as _os
import logging as _logging
import scipy as _scipy
import scipy.misc
import glob as _glob

def draw_stickfigure(image, pose, circlesize=4, stickwidth=4, debug=False):
    '''Visualize estimated 2D joints as stick figures. Note that the input and output image channels are BGR'''
    # joint names and index
    # 0 Right ankle
    # 1 Right knee
    # 2 Right hip
    # 3 Left hip
    # 4 Left knee
    # 5 Left ankle
    # 6 Right wrist
    # 7 Right elbow
    # 8 Right shoulder
    # 9 Left shoulder
    # 10 Left elbow
    # 11 Left wrist
    # 12 Neck
    # 13 Head top

    '''colors_rgb = [[0, 0, 255],[51,153,255],[102,255,255], \
              [127,0,255], [255,51,255], [255,102,178], \
              [0,255,0], [153,255,51], [102,255,178], \
              [204,255,153], [255,255,51], [255,128,0], \
              [255,0,0], [255,204,204]]'''
    # we use BGR color
    colors = [[255,0,0],[255,153,51],[255,255,102], \
              [255,0,127], [255,51,255], [178,102,255], \
              [0,255,0], [51,255,153], [178,255,102], \
              [0,128,255], [51,255,255], [153,255,204], \
              [0,0,255], [204,204,255]]

    #low_conf_color = [100,100,100]
    #very_low_conf_color = [0,0,0]

    # find connection in the specified sequence
    limb_seq = [[0,1], [1,2], [2, 12], [4,5], [3,4], [3,12], [6,7], [7,8], [8,12], \
               [10,11], [9,10], [9,12], [12,13]]

    visim = image.copy()
    joint_seq = [0,1,2,5,4,3,6,7,8,11,10,9,13,12]
    for i in range(len(joint_seq)):
        # draw circle
        cur_visim = visim.copy()
        idx = joint_seq[i]
        center = (int(pose[0,idx]), int(pose[1,idx]))
        if 0.88 < pose[2,idx] <= 0.92:
            cv2.circle(cur_visim, center, 6, colors[i], thickness=-1)
        elif pose[2,idx] <= 0.88:
            cv2.circle(cur_visim, center, 4, colors[i], thickness=-1)
        else:
            cv2.circle(cur_visim, center, 8, colors[i], thickness=-1)
        visim = cv2.addWeighted(visim, 0.25, cur_visim, 0.75, 0)

        if debug:
            import matplotlib.pyplot as plt
            plt.ion()
            #plt.show()
            plt.subplot(131)
            plt.imshow(image[:,:,[2,1,0]])
            plt.subplot(132)
            plt.imshow(visim[:,:,[2,1,0]])
            plt.subplot(133)
            plt.imshow(cur_visim[:,:,[2,1,0]])
            plt.cla()
            plt.show()
            plt.pause(1)
            raw_input('Press any key to continue...')

        if i < 13:
            # draw sticke
            cur_visim = visim.copy()
            index = np.array(limb_seq[i])
            X = pose[0, index.astype(int)]
            Y = pose[1, index.astype(int)]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygon = cv2.ellipse2Poly((int(mX),int(mY)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_visim, polygon, colors[i])
            visim = cv2.addWeighted(visim, 0.4, cur_visim, 0.6, 0)

    return visim

def main(image_name, joint_dir, out_dir, folder_image_suffix):
    """
    given images and the corresponding 2D joint positions (stored in .npz),
    output images with stick figure visualizations
    """
    if _os.path.isdir(image_name):
        folder_name = image_name[:]
        _LOGGER.info("Specified image name is a folder. Processing all images "
                     "with suffix %s.", folder_image_suffix)
        print 'path to images: '+ folder_name
        images = sorted(_glob.glob(_os.path.join(folder_name, '*' + folder_image_suffix)))

        print 'path to 2D joints: '+joint_dir
        joints = sorted(_glob.glob(_os.path.join(joint_dir, '*[0-9]*.npz')))

        process_folder = True
    else:
        images = [image_name]
        process_folder = False

    if process_folder and out_dir is not None and not _os.path.exists(out_dir):
        _os.mkdir(out_dir)
    #else:
        #_os.mkdir('/sequoia/data2/zoli/datazml/deepercut_tmp')
    for ite, image_name in enumerate(images):
        print 'iteration ' + str(ite)

        # load 2d joints
        joint_path = joints[ite]
        est = np.load(joint_path)['pose']
        pose = est[:3, :]

        # set output parameters
        image_name = image_name[:-4]
        if out_dir is None:
            out_name = _os.path.join('/sequoia/data2/zoli/datazml/deepercut_tmp', _os.path.basename(image_name) + '_pose_vis.png')
        elif process_folder:
            out_name = _os.path.join(out_dir, _os.path.basename(image_name) + '_pose_vis.png')
        else:
            out_name = out_dir+'_pose_vis.png'
        _LOGGER.info("Generating stickfigures on `%s` (saving to `%s`)", _os.path.basename(image_name)+folder_image_suffix, out_dir)

        image = _scipy.misc.imread(image_name+folder_image_suffix) # in RGB order
        if image.ndim == 2:
            _LOGGER.warn("The image is grayscale! This may deteriorate performance!")
            image = _np.dstack((image, image, image))
        else:
            image = image[:, :, ::-1] # convert to BGR order

        debug = False
        circlesize = 4
        stickwidth = 4
        visim = draw_stickfigure(image, pose, circlesize, stickwidth, debug)
        _scipy.misc.imsave(out_name, visim[:,:,::-1])

if __name__ == '__main__':
    import argparse
    _LOGGER = _logging.getLogger(__name__)
    _logging.basicConfig(level=_logging.INFO)
    parser = argparse.ArgumentParser(description='run SMPLify on LSP dataset')
    parser.add_argument(
        'image_name',
        default='/sequoia/data2/zoli/datazml/ins_video_frames/changing_tire_0015/remove_wheel',
        nargs='?',
        help="Directory that contains images")
    parser.add_argument(
        'joint_dir',
        default='/sequoia/data2/zoli/datazml/ins_video_pose/changing_tire_0015/remove_wheel',
        nargs='?',
        help="Directory that contains 2d joint results")
    parser.add_argument(
        'out_dir',
        default='/sequoia/data2/zoli/datazml/ins_video_pose_stickfigure/changing_tire_0015/remove_wheel',
        nargs='?',
        help='Where results will be saved')
    parser.add_argument(
        '--folder_image_suffix',
        default='.jpg', #or neutral or female
        type=str,
        help='specify image suffix, default is .jpg')
    args = parser.parse_args()

    main(args.image_name, args.joint_dir, args.out_dir, args.folder_image_suffix)
