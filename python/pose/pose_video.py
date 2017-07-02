#!/usr/bin/env python
"""
Given a video and the starting and ending frames, estimate 2D joints using DeeperCut.
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

_LOGGER = logging.getLogger(__name__)

def predict_j2d(video_path,save_dir,frames,scales,viz,use_cpu,gpu):
    """
    Load a video file, predict the pose and write it out.
    """
    model_def = '/sequoia/data2/zoli/deepcut-cnn/models/deepercut/ResNet-152.prototxt'
    model_bin = '/sequoia/data2/zoli/deepcut-cnn/models/deepercut/ResNet-152.caffemodel'

    frames = [int(val) for val in frames.split(',')]
    scales = [float(val) for val in scales.split(',')]
    video_name = basename(video_path)

    _LOGGER.info("Predicting the pose on '%s'",video_name)
    _LOGGER.info("path to video %s",video_path)
    _LOGGER.info("saving to folder %s",save_dir)
    _LOGGER.info("selected scales %s",scales)

    if use_cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu)

    for sq in range(len(frames)/2):
        frame0 = frames[sq*2]
        frame1 = frames[sq*2+1]

        out_name = video_name[:-4]+'_f{0}-{1}_j2d.npz'.format(frame0,frame1)
        out_path = join(save_dir, out_name)

        cap = cv2.VideoCapture(video_path)
        frameMax = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame0 = frameMax if frame0>frameMax else frame0
        frame1 = frameMax if frame1>frameMax else frame1
        _LOGGER.info("processing from frame %s to frame %s",frame0,frame1)

        cap.set(1, frame0)

        if viz:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            fps = cap.get(cv2.CAP_PROP_FPS)
            fwidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            viz_path = out_path[:-4] + '_viz.mp4'
            print type(viz_path),type(fourcc),type(fps),type(fwidth),type(fheight)
            print viz_path,fourcc,fps,fwidth,fheight
            out_viz = cv2.VideoWriter(viz_path, fourcc, fps, (fwidth,fheight))

        j2d_all = []
        best_scales = []
        fIdx = frame0
        while(cap.isOpened() and frame0<=fIdx<=frame1):
            ret, image = cap.read()
            if image.ndim == 2:
                _LOGGER.warn("Frame is in grayscale!")
                image = np.dstack((image, image, image))
            pose, best_scale = estimate_pose(image, model_def, model_bin, scales)

            _LOGGER.info("Frame %s. The best scale is %s.", fIdx, best_scale)
            j2d_all.append(pose)
            best_scales.append(best_scale)

            if viz:
                debug = False
                circlesize = 4
                stickwidth = 4
                visim = draw_stickfigure(image, pose, circlesize, stickwidth, debug)
                out_viz.write(visim)
            fIdx += 1

        # Release everything if job is finished
        cap.release()
        if viz:
            out_viz.release()
        cv2.destroyAllWindows()

        # save estimated j2d pose in save_dir
        j2d_all = np.array(j2d_all)
        best_scales = np.array(best_scales)
        np.savez_compressed(out_path, j2d_all=j2d_all,
                                      frames=frames,
                                      scales_tried=scales,
                                      best_scales=best_scales)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='run DeeperCut on a video (sequence of frames)')
    parser.add_argument(
        'video_path',
        default='/sequoia/data2/zoli/datapublic/ins_videos/changing_tire/videos/changing_tire_0006.mpg',
        nargs='?',
        help="Path to input video")
    parser.add_argument(
        '--save_dir',
        default='/sequoia/data2/zoli/datazml/ins_video/changing_tire_0006/',
        nargs='?',
        help="Output directory")
    parser.add_argument(
        '--frames',
        default='0,10',
        type=str,
        help="a list containing all frame indices to process")
    parser.add_argument(
        '--scales',
        default='1.',
        type=str,
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
    predict_j2d(args.video_path, args.save_dir, args.frames,
                args.scales, args.viz, args.use_cpu, args.gpu)
