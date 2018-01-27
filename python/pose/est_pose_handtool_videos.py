import logging as logging
import numpy as np
import caffe
import argparse
import cv2 as cv
from os.path import join, exists, abspath, dirname, basename
from estimate_pose import estimate_pose
from visualize import draw_stickfigure
import cPickle as pickle

_LOGGER = logging.getLogger(__name__)

def npcircle(image, cx, cy, radius, color, transparency=0.0):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    if cy<radius:
        cy = radius
    elif cy>image.shape[0]-radius:
        cy = image.shape[0]-radius
    if cx<radius:
        cx = radius
    elif cx>image.shape[1]-radius:
        cx = image.shape[1]-radius
    image[cy-radius:cy+radius, cx-radius:cx+radius][index] = (
        image[cy-radius:cy+radius, cx-radius:cx+radius][index].astype('float32') * transparency +
        np.array(color).astype('float32') * (1.0 - transparency)).astype('uint8')

def run_deepercut(video_name, frames_dir, out_dir, info_path, scale_guess):
    GPUdeviceNumber = 0
    ###### load video info
    with open(info_path, 'r') as finfo:
        info = pickle.load(finfo)
        nframes = int(info[video_name]['nframes'])
        fwidth = int(info[video_name]['fwidth']) # frame width
        fheight = int(info[video_name]['fheight']) # frame height
        fps = info[video_name]['fps'] # frame per second

    scaleMin = scale_guess-0.26 if scale_guess-0.25>0. else 0.1
    scaleMax = scale_guess+0.26
    scales = np.linspace(scaleMin, scaleMax, num=20)

    ##### load caffe model
    model_def = '/sequoia/data2/zoli/deepcut-cnn/models/deepercut/ResNet-152.prototxt'
    model_bin = '/sequoia/data2/zoli/deepcut-cnn/models/deepercut/ResNet-152.caffemodel'
    caffe.set_mode_gpu()
    caffe.set_device(GPUdeviceNumber)

    joints_2d = []
    for fid in range(nframes):
        test_image = join(frames_dir, video_name, "%06d.jpg" % fid)
        _LOGGER.info("processing"+test_image)
        image = cv.imread(test_image)

        if image.ndim == 2:
            _LOGGER.warn("Frame is in grayscale!")
            image = np.dstack((image, image, image))
        pose, best_scale = estimate_pose(image, model_def, model_bin, scales)

        _LOGGER.info("fid %s. Best scale is %s.", fid, best_scale)
        joints_2d.append(pose)

        circlesize = 4
        stickwidth = 2
        visim = draw_stickfigure(image, pose, circlesize, stickwidth, False)
        save_path = join(out_dir, video_name, "%06d.jpg" % fid)
        cv.imwrite(save_path, visim)

    # save estimated j2d pose in save_dir
    joints_2d = np.array(joints_2d)

    # dave result to pkl
    data_path = join(out_dir, 'deepercut.pkl')
    if exists(data_path):
        with open(data_path, 'r') as fdata:
            deepercut_res = pickle.load(fdata)
            deepercut_res[video_name] = joints_2d
    else:
        deepercut_res = {video_name:joints_2d}

    with open(data_path, 'w') as fout:
        pickle.dump(deepercut_res, fout)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='run DeeperCut on a video (sequence of frames)')

    parser.add_argument(
        'video_name',
        nargs='?',
        default='pitchfork_0006',
        help="name of video")
    parser.add_argument(
        'frames_dir',
        nargs='?',
        default='/sequoia/data2/zoli/100_handtool_videos/pitchfork/frames',
        help="path to video frames")
    parser.add_argument(
        'out_dir',
        nargs='?',
        default='/sequoia/data2/zoli/100_handtool_videos/pitchfork/deepercut',
        help="output directory")
    parser.add_argument(
        'info_path',
        nargs='?',
        default='/sequoia/data2/zoli/100_handtool_videos/pitchfork/videos_info.pkl',
        help="path to video info")
    parser.add_argument(
        '--scale_guess',
        default=1.,
        type=float,
        help="The scales to use, comma-separated. The most confident will be stored.")
    args = parser.parse_args()
    run_deepercut(args.video_name,
                  args.frames_dir,
                  args.out_dir,
                  args.info_path,
                  args.scale_guess)
