#!/usr/bin/env python
"""
Pose predictions in Python.

Caffe must be available on the Pythonpath for this to work. The methods can
be imported and used directly, or the command line interface can be used. In
the latter case, adjust the log-level to your needs. The maximum image size
for one prediction can be adjusted with the variable _MAX_SIZE so that it
still fits in GPU memory, all larger images are split in sufficiently small
parts.

Authors: Christoph Lassner, based on the MATLAB implementation by Eldar
  Insafutdinov.
"""
# pylint: disable=invalid-name
import os as _os
import logging as _logging
import glob as _glob
import numpy as _np
import scipy as _scipy
import click as _click
import caffe as _caffe

from estimate_pose import estimate_pose
from visualize import draw_stickfigure

_LOGGER = _logging.getLogger(__name__)

def _npcircle(image, cx, cy, radius, color, transparency=0.0):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    y, x = _np.ogrid[-radius: radius, -radius: radius]
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
        _np.array(color).astype('float32') * (1.0 - transparency)).astype('uint8')


###############################################################################
# Command line interface.
###############################################################################

@_click.command()
@_click.argument('image_name',
                 type=_click.Path(exists=True, dir_okay=True, readable=True))
@_click.option('--out_name',
               type=_click.Path(dir_okay=True, writable=True),
               help='The result location to use. By default, use `image_name`_pose.npz.', # out_name should be a path to folder
               default=None)
@_click.option('--scales',
               type=_click.STRING,
               help=('The scales to use, comma-separated. The most confident '
                     'will be stored. Default: 1.'),
               default='1.')
@_click.option('--visualize',
               type=_click.BOOL,
               help='Whether to create a visualization of the pose. Default: True.',
               default=True)
@_click.option('--folder_image_suffix',
               type=_click.STRING,
               help=('The ending to use for the images to read, if a folder is '
                     'specified. Default: .jpg.'),
               default='.jpg')
@_click.option('--use_cpu',
               type=_click.BOOL,
               is_flag=True,
               help='Use CPU instead of GPU for predictions.',
               default=False)
@_click.option('--gpu',
               type=_click.INT,
               help='GPU device id.',
               default=0)
def predict_pose_from(image_name,
                      out_name=None,
                      scales='1,',
                      visualize=True,
                      folder_image_suffix='.jpg',
                      use_cpu=False,
                      gpu=0):
    """
    Load an image file, predict the pose and write it out.

    `IMAGE_NAME` may be an image or a directory, for which all images with
    `folder_image_suffix` will be processed.
    """
    model_def = '/sequoia/data2/zoli/deepcut-cnn/models/deepercut/ResNet-152.prototxt'
    model_bin = '/sequoia/data2/zoli/deepcut-cnn/models/deepercut/ResNet-152.caffemodel'
    scales = [float(val) for val in scales.split(',')]
    print scales
    if _os.path.isdir(image_name):
        folder_name = image_name[:]
        _LOGGER.info("Specified image name is a folder. Processing all images "
                     "with suffix %s.", folder_image_suffix)
        images = _glob.glob(_os.path.join(folder_name, '*' + folder_image_suffix))
        process_folder = True
    else:
        images = [image_name]
        process_folder = False
    if use_cpu:
        _caffe.set_mode_cpu()
    else:
        _caffe.set_mode_gpu()
        _caffe.set_device(gpu)
    out_name_provided = out_name
    if process_folder and out_name is not None and not _os.path.exists(out_name):
        _os.mkdir(out_name)
    for image_name in images:
        image_name = image_name[:-4]
        if out_name_provided is None:
            out_name = _os.path.join('/sequoia/data2/zoli/datazml/deepercut_tmp', _os.path.basename(image_name) + '_j2d.npz')
        elif process_folder:
            out_name = _os.path.join(out_name_provided, _os.path.basename(image_name) + '_j2d.npz')
        else:
            out_name = out_name+'_j2d.npz'
        _LOGGER.info("Predicting the pose on `%s` (saving to `%s`) in best of "
                     "scales %s.", image_name+folder_image_suffix, out_name_provided, scales)

        image = _scipy.misc.imread(image_name+folder_image_suffix) # in RGB order
        if image.ndim == 2:
            _LOGGER.warn("The image is grayscale! This may deteriorate performance!")
            image = _np.dstack((image, image, image))
        else:
            image = image[:, :, ::-1] # convert to BGR order
        pose, best_scale = estimate_pose(image, model_def, model_bin, scales)
        _LOGGER.info("The best scale is %s.", best_scale)
        _np.savez_compressed(out_name, pose=pose, scale=best_scale)
        #####
        if visualize:
            '''
            visim = image[:, :, ::-1].copy() # conver to RGB order
            colors = [[255, 0, 0],[0, 255, 0],[0, 0, 255],[0,245,255],[255,131,250],[255,255,0],
                      [255, 0, 0],[0, 255, 0],[0, 0, 255],[0,245,255],[255,131,250],[255,255,0],
                      [0,0,0],[255,255,255]]
            for p_idx in range(14):
                _npcircle(visim,
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
            vis_name = out_name[:-4] + '_vis.png'
            _scipy.misc.imsave(vis_name, visim[:,:,::-1])

if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO)
    # pylint: disable=no-value-for-parameter
    predict_pose_from()
