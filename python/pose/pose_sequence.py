#!/usr/bin/env python
"""
Given a sequence of frames, estimate 2D joints positions using DeeperCut.
"""
# pylint: disable=invalid-name
import logging as logging
import numpy as np
import caffe
import argparse
import cv2
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
    #print image.shape
    #print cy-radius
    #print cy+radius
    #print cx-radius
    #print cx+radius
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

def predict_j2d(video_name,db_dir,frames,scale_guess,viz,use_cpu,gpu):
    """
    Predict the pose and write it out.
    """

    model_def = '/sequoia/data2/zoli/deepcut-cnn/models/deepercut/ResNet-152.prototxt'
    model_bin = '/sequoia/data2/zoli/deepcut-cnn/models/deepercut/ResNet-152.caffemodel'

    seq_name = video_name.split('.')[0]
    viz_dir = join(db_dir, seq_name, 'viz')
    image_dir = join(viz_dir, 'frames')
    save_dir = join(db_dir, seq_name, 'data', 'deepercut')
    info_path = join(db_dir, seq_name, 'data', 'video_info.pkl')

    frames = [int(val) for val in frames.split(',')]
    #scales = [float(val) for val in scales.split(',')]
    scaleMin = scale_guess-0.26 if scale_guess-0.25>0. else 0.1
    scaleMax = scale_guess+0.26
    scales = np.linspace(scaleMin, scaleMax, num=27)

    _LOGGER.info("working under folder %s",image_dir)
    _LOGGER.info("frames %s",frames)
    _LOGGER.info("saving to folder %s",save_dir)
    _LOGGER.info("center scale %s",scale_guess)

    if use_cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu)

    with open(info_path, 'r') as finfo:
        video_info = pickle.load(finfo)
    fps = video_info['fps'] # frame per second
    nframes = int(video_info['nframes']) # total number of frames
    fwidth = int(video_info['fwidth']) # frame width
    fheight = int(video_info['fheight']) # frame height

    for sq in range(len(frames)/2):
        frame0 = frames[sq*2]
        frame1 = frames[sq*2+1]

        out_name = seq_name+'_f{0}-{1}_j2d.pkl'.format(frame0,frame1)
        out_path = join(save_dir, out_name)

        frame0 = nframes if frame0>nframes else frame0
        frame1 = nframes if frame1>nframes else frame1
        _LOGGER.info("processing from frame %s to frame %s",frame0,frame1)

        if viz:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            viz_path = join(viz_dir, 'deepercut', out_name.split('.')[0] + '_viz.mp4')
            out_viz = cv2.VideoWriter(viz_path, fourcc, fps, (fwidth,fheight))

        #raw_input('waiting key...')

        j2d_seq = []
        best_scales = []
        fIdx = frame0
        while(frame0<=fIdx<=frame1):
            image_path = join(image_dir, "%06d.jpg" % fIdx)
            image = cv2.imread(image_path)

            if image.ndim == 2:
                _LOGGER.warn("Frame is in grayscale!")
                image = np.dstack((image, image, image))
            pose, best_scale = estimate_pose(image, model_def, model_bin, scales)

            _LOGGER.info("Frame %s. The best scale is %s.", fIdx, best_scale)
            j2d_seq.append(pose)
            best_scales.append(best_scale)

            if viz:
                '''
                visim = image[:, :, ::-1].copy() # conver to RGB order
                colors = [[255, 0, 0],[0, 255, 0],[0, 0, 255],[0,245,255],[255,131,250],[255,255,0],
                          [255, 0, 0],[0, 255, 0],[0, 0, 255],[0,245,255],[255,131,250],[255,255,0],
                          [0,0,0],[255,255,255]]
                for p_idx in range(14):
                    npcircle(visim,
                              pose[0, p_idx],
                              pose[1, p_idx],
                              4,
                              colors[p_idx],
                              0.0)
                '''
                debug = False
                circlesize = 4
                stickwidth = 4
                visim = draw_stickfigure(image, pose, circlesize, stickwidth, debug)

                out_viz.write(visim)
            fIdx += 1

        # Release everything if job is finished
        if viz:
            out_viz.release()
        cv2.destroyAllWindows()

        # save estimated j2d pose in save_dir
        j2d_seq = np.array(j2d_seq)
        best_scales = np.array(best_scales)

        params = {'j2d_seq': j2d_seq,
                  'frames': frames,
                  'scales': scales,
                  'best_scales': best_scales}

        with open(out_path, 'w') as outf:
            pickle.dump(params, outf)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='run DeeperCut on a video (sequence of frames)')
    parser.add_argument(
        'video_name',
        default='changing_tire_0003.mpg',
        nargs='?',
        help="name of the video, with extensions (changing_tire_0003.mpg by default)")
    parser.add_argument(
        '--db_dir',
        default='/sequoia/data2/zoli/tool_videos',
        nargs='?',
        help="Path to data directory")
    parser.add_argument(
        '--frames',
        default='0,10',
        type=str,
        help="a list containing all frame indices to process")
    parser.add_argument(
        '--scale_guess',
        default=1.,
        type=float,
        help="The scales to use, comma-separated. The most confident will be stored.")
    parser.add_argument(
        '--viz',
        default=True,
        action='store_false',
        help="save estimated 2D joints with stickfigures as a new video")
    parser.add_argument(
        '--use_cpu',
        default=False,
        action='store_true',
        help="Use CPU instead of GPU for predictions.")
    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help="GPU device id.")
    args = parser.parse_args()

    predict_j2d(args.video_name, args.db_dir, args.frames,
                args.scale_guess, args.viz, args.use_cpu, args.gpu)
