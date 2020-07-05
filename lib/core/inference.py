# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds
import cv2

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


def _extract_points_from_heatmaps(heatmaps, tag_maps, thresh=0.95):
    points_all = []
    for _, heatmaps_per_bbox, tags_per_bbox in enumerate(zip(heatmaps, tag_maps)):
        points = []
        for i in range(len(heatmaps_per_bbox)):
            locs = np.where(heatmaps_per_bbox[i] >= thresh)
            if len(locs[0])==0:
                locs = np.where(heatmaps_per_bbox[i] >= heatmaps_per_bbox[i].max())
            scores = heatmaps_per_bbox[i][locs]
            tag_values = tags_per_bbox[locs]
            locs = np.asarray(locs).transpose()
            ind = np.argsort(scores)[::-1]
            scores = scores[ind]
            tag_values = tag_values[ind]
            locs = locs[ind]
            dist = compute_points_dist(locs)
            remove_tags = np.zeros((len(scores)))
            target_points = []
            for j in range(len(scores)):
                if remove_tags[j] == 1:
                    continue
                target_points += [locs[j, 0], locs[j, 1], i, scores[j], tag_values[j]]
                current_score = scores[j]
                remove_cands = dist[j].flatten()
                remove_targets = np.logical_and(remove_cands < 3, scores <= current_score)
                remove_targets[j] = 0
                remove_tags[remove_targets] = 1
            points.append(np.asarray(target_points).reshape((-1,5)))
        target_points_per_person = []

        for i in range(len(points)):

            tags_per_person = np.asarray(tags_per_person)


        if len(points)>0:
            points_all.append(points)
        else:
            points_all.append([])
    return points_all

def compute_points_dist(points):
    loc_x = points[:,0].reshape((-1,1))
    loc_y = points[:,1].reshape((-1,1))
    dx = (loc_x[:,None] - loc_x)**2
    dy = (loc_y[:,None] - loc_y)**2
    dist = np.sqrt(dx + dy)
    return dist


