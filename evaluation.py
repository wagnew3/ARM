import sys, os
import numpy as np
import cv2

import util.munkres as munkres
import util.util as util_

BACKGROUND_LABEL = 0
TABLE_LABEL = 1
OBJECTS_LABEL = 2

def IoU(mask1, mask2):
    """ Calculates IoU of two segmentation masks

        @param mask1: a [H x W] numpy array of 0s and 1s
        @param mask2: a [H x W] numpy array of 0s and 1s

        @return: a [batch_size] array of IoUs 
    """

    # Turn them into boolean arrays
    if not mask1.dtype == bool:
        mask1 = mask1.astype(bool)
    if not mask2.dtype == bool:
        mask2 = mask2.astype(bool)

    # Calculate number of pixels in intersection
    intersection = np.logical_and(mask1, mask2)

    # Calculate number of pixels in union
    union = np.logical_or(mask1, mask2)

    # Return the IoU
    return np.sum(intersection) / (np.sum(union) + 1e-10)

def batch_IoU(mask1, mask2):
    """ Calculates IoU of two segmentation masks

        @param mask1: a [batch_size x H x W] numpy array of 0s and 1s
        @param mask2: a [batch_size x H x W] numpy array of 0s and 1s

        @return: a [batch_size] array of IoUs 
    """

    # Turn them into boolean arrays
    if not mask1.dtype == bool:
        mask1 = mask1.astype(bool)
    if not mask2.dtype == bool:
        mask2 = mask2.astype(bool)

    # Calculate number of pixels in intersection
    intersection = np.logical_and(mask1, mask2)

    # Calculate number of pixels in union
    union = np.logical_or(mask1, mask2)

    # Return the mean IoU
    temp = np.sum(intersection, axis=(1,2)) / (np.sum(union, axis=(1,2)) + 1e-10)
    return temp

def avg_IoU(mask1, mask2):
    """ Calculates IoU of two sets of segmentation masks of sequences

        @param mask1: a [batch_size x H x W] numpy array of 0s and 1s
        @param mask2: a [batch_size x H x W] numpy array of 0s and 1s
    """

    return np.mean(batch_IoU(mask1, mask2))


# Code adapted from: https://github.com/davisvideochallenge/davis-2017/blob/master/python/lib/davis/measures/f_boundary.py
def seg2bmap(seg,width=None,height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width     : Width of desired bmap  <= seg.shape[1]
        height  :   Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray): Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
 """

    seg = seg.astype(np.bool)
    seg[seg>0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width  = seg.shape[1] if width  is None else width
    height = seg.shape[0] if height is None else height

    h,w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
            'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e  = np.zeros_like(seg)
    s  = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:,:-1]    = seg[:,1:]
    s[:-1,:]    = seg[1:,:]
    se[:-1,:-1] = seg[1:,1:]

    b        = seg^e | seg^s | seg^se
    b[-1,:]  = seg[-1,:]^e[-1,:]
    b[:,-1]  = seg[:,-1]^s[:,-1]
    b[-1,-1] = 0

    # from IPython import embed; embed()

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height,width))
        for x in range(w):
            for y in range(h):
                if b[y,x]:
                    j = 1+floor((y-1)+height / h)
                    i = 1+floor((x-1)+width  / h)
                    bmap[j,i] = 1;

    return bmap

# Code adapted from: https://github.com/davisvideochallenge/davis-2017/blob/master/python/lib/davis/measures/f_boundary.py
def boundary_overlap(predicted_mask, gt_mask, bound_th=0.003):
    """
    Compute IoU of boundaries of GT/predicted mask, using dilated GT boundary
    Arguments:
        predicted_mask  (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
    Returns:
        overlap (float): IoU overlap of boundaries
    """
    assert np.atleast_3d(predicted_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(predicted_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(predicted_mask);
    gt_boundary = seg2bmap(gt_mask);

    from skimage.morphology import disk

    # Dilate segmentation boundaries
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix), iterations=1)
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix), iterations=1)

    # Get the intersection (true positives). Calculate true positives differently for
    #   precision and recall since we have to dilated the boundaries
    fg_match = np.logical_and(fg_boundary, gt_dil)
    gt_match = np.logical_and(gt_boundary, fg_dil)

    # Return precision_tps, recall_tps (tps = true positives)
    return np.sum(fg_match), np.sum(gt_match)

# This function is modeled off of P/R/F measure as described by Dave et al. (arXiv19)
def multilabel_metrics(prediction, gt, obj_detect_threshold=0.75):
    """ Computes F-Measure, Precision, Recall, IoU, #objects detected, #confident objects detected, #GT objects.
        It computes these measures only of objects, not background (0)/table (1).
        Uses the Hungarian algorithm to match predicted masks with ground truth masks.

        A "confident object" is an object that is predicted with more than 0.75 F-measure

        @param gt: a [H x W] numpy.ndarray with ground truth masks
        @param prediction: a [H x W] numpy.ndarray with predicted masks

        @return: a dictionary with the metrics
    """

    ### Compute F, TP matrices ###

    # Get unique OBJECT labels from GT and prediction
    labels_gt = np.unique(gt)
    labels_gt = labels_gt[~np.isin(labels_gt, [BACKGROUND_LABEL, TABLE_LABEL])]
    num_labels_gt = labels_gt.shape[0]

    labels_pred = np.unique(prediction)
    labels_pred = labels_pred[~np.isin(labels_pred, [BACKGROUND_LABEL, TABLE_LABEL])]
    num_labels_pred = labels_pred.shape[0]

    # F-measure, True Positives, Boundary stuff
    F = np.zeros((num_labels_gt, num_labels_pred))
    true_positives = np.zeros((num_labels_gt, num_labels_pred))
    boundary_stuff = np.zeros((num_labels_gt, num_labels_pred, 2)) 
    # Each item of "coundary_stuff" contains: precision_tps, recall_tps

    # Some edge case stuff
    # Edge cases are similar to here: https://github.com/davisvideochallenge/davis-2017/blob/master/python/lib/davis/measures/f_boundary.py
    if (num_labels_pred == 0 and num_labels_gt > 0 ): # all false negatives
        return {'Objects F-measure' : 0.,
                'Objects Precision' : 1.,
                'Objects Recall' : 0.,
                'Boundary F-measure' : 0.,
                'Boundary Precision' : 1.,
                'Boundary Recall' : 0.,
                'obj_detected' : num_labels_pred,
                'obj_detected_075' : 0.,
                'obj_gt' : num_labels_gt,
                'obj_detected_075_percentage' : 0.,
                }
    elif (num_labels_pred > 0 and num_labels_gt == 0 ): # all false positives
        return {'Objects F-measure' : 0.,
                'Objects Precision' : 0.,
                'Objects Recall' : 1.,
                'Boundary F-measure' : 0.,
                'Boundary Precision' : 0.,
                'Boundary Recall' : 1.,
                'obj_detected' : num_labels_pred,
                'obj_detected_075' : 0.,
                'obj_gt' : num_labels_gt,
                'obj_detected_075_percentage' : 0.,
                }
    elif (num_labels_pred == 0 and num_labels_gt == 0 ): # correctly predicted nothing
        return {'Objects F-measure' : 1.,
                'Objects Precision' : 1.,
                'Objects Recall' : 1.,
                'Boundary F-measure' : 1.,
                'Boundary Precision' : 1.,
                'Boundary Recall' : 1.,
                'obj_detected' : num_labels_pred,
                'obj_detected_075' : 0.,
                'obj_gt' : num_labels_gt,
                'obj_detected_075_percentage' : 1.,
                }

    # For every pair of GT label vs. predicted label, calculate stuff
    for i, gt_i in enumerate(labels_gt):

        gt_i_mask = (gt == gt_i)

        for j, pred_j in enumerate(labels_pred):
            
            pred_j_mask = (prediction == pred_j)
            
            ### Overlap Stuff ###

            # true positive
            A = np.logical_and(pred_j_mask, gt_i_mask)
            tp = np.int64(np.count_nonzero(A)) # Cast this to numpy.int64 so 0/0 = nan
            true_positives[i,j] = tp 
            
            # precision
            prec = tp/np.count_nonzero(pred_j_mask)
            
            # recall
            rec = tp/np.count_nonzero(gt_i_mask)
            
            # F-measure
            F[i,j] = (2 * prec * rec) / (prec + rec)

            ### Boundary Stuff ###
            boundary_stuff[i,j] = boundary_overlap(pred_j_mask, gt_i_mask)

    ### More Boundary Stuff ###
    boundary_prec_denom = 0. # precision_tps + precision_fps
    for pred_j in labels_pred:
        pred_mask = (prediction == pred_j)
        boundary_prec_denom += np.sum(seg2bmap(pred_mask))
    boundary_rec_denom = 0. # recall_tps + recall_fns
    for gt_i in labels_gt:
        gt_mask = (gt == gt_i)
        boundary_rec_denom += np.sum(seg2bmap(gt_mask))


    ### Compute the Hungarian assignment ###
    F[np.isnan(F)] = 0
    m = munkres.Munkres()
    assignments = m.compute(F.max() - F.copy()) # list of (y,x) indices into F (these are the matchings)

    ### Compute the number of "detected objects" ###
    num_obj_detected = 0
    for a in assignments:
        if F[a] > obj_detect_threshold:
            num_obj_detected += 1

    ### Compute metrics with assignments ###
    idx = tuple(np.array(assignments).T)

    # Overlap measures
    precision = np.sum(true_positives[idx]) / np.sum(prediction.clip(0,2) == OBJECTS_LABEL)
    recall = np.sum(true_positives[idx]) / np.sum(gt.clip(0,2) == OBJECTS_LABEL)
    F_measure = (2 * precision * recall) / (precision + recall)
    if np.isnan(F_measure): # b/c precision = recall = 0
        F_measure = 0

    # Boundary measures
    boundary_precision = np.sum(boundary_stuff[idx][:,0]) / boundary_prec_denom
    boundary_recall = np.sum(boundary_stuff[idx][:,1]) / boundary_rec_denom
    boundary_F_measure = (2 * boundary_precision * boundary_recall) / (boundary_precision + boundary_recall)
    if np.isnan(boundary_F_measure): # b/c/ precision = recall = 0
        boundary_F_measure = 0


    return {'Objects F-measure' : F_measure,
            'Objects Precision' : precision,
            'Objects Recall' : recall,
            'Boundary F-measure' : boundary_F_measure,
            'Boundary Precision' : boundary_precision,
            'Boundary Recall' : boundary_recall,
            'obj_detected' : num_labels_pred,
            'obj_detected_075' : num_obj_detected,
            'obj_gt' : num_labels_gt,
            'obj_detected_075_percentage' : num_obj_detected / num_labels_gt,
            }

def all_combos(list_of_lists):
    """ Compute all combos of strings given in list. Separate with underscores

        Example:
            list_of_lists = [['fish', 'dog'], ['hat', 'pants']]

        Outpu:
            ['fish', 'fish_hat', 'fish_pants',
             'dog',  'dog_hat',  'dog_pants'
            ]
    """

    combos = []
    recent_combos = ['']

    for _list in list_of_lists:
        new_combos = []

        for elem in _list:
            for combo in recent_combos:

                new_elem = combo + '_' + elem
                if new_elem.startswith('_'):
                    new_elem = new_elem[1:]

                new_combos.append(new_elem)

        recent_combos = new_combos
        combos = combos + new_combos

    return combos

def evaluate_on_dataset(prediction_dir, gt_dir, dataset_name):
    """ Compute metrics on a dataset 

        @param gt_dir: string. directory to GT annotations. This might be a tree of subdirectories
        @param prediction_dir: string. directory to predicted labels. The directory structure should MATCH the gt_dir
        @param dataset_name: a string in ['OCID', 'OSD', 'RGBO', 'Synth']
    """
    from tqdm import tqdm_notebook as tqdm # Because i'm using jupyter notebook. This can be something else for command line usage

    if dataset_name == 'OCID' or dataset_name == 'OSD':
        starts_file = gt_dir + 'label_files.txt'
    elif dataset_name == 'RGBO':
        pass # TODO: implement this
    elif dataset_name == 'Synth':
        pass # TODO: implement this. also name it something better than "Synth"...

    # Parse starts file
    f = open(starts_file, 'r')
    label_filenames = [x.strip() for x in f.readlines()]


    metrics = {} # a dictionary of label_filename -> dictionary of metrics
    if dataset_name == 'OCID': # For OCID, keep track of different combos. hard code this. yeah it's ugly
        OCID_subset_metrics = {}

        OCID_subsets = all_combos([['ARID20','ARID10','YCB10'],['table','floor'],['top','bottom']])

        for subset in OCID_subsets:
            OCID_subset_metrics[subset] = {}

    for label_filename in tqdm(label_filenames):

        # Load the GT label file
        if dataset_name == 'OCID':

            temp_idx = label_filename.split('/').index('OCID-dataset') # parse something like this: /data/OCID-dataset/YCB10/table/top/curved/seq36/pcd/result_2018-08-24-15-13-13.pcd
            subset = label_filename.split('/')[temp_idx+1] # one of: [ARID10, ARID20, YCB10]
            supporting_plane = 'floor' if 'floor' in label_filename else 'table'
            height = 'bottom' if 'bottom' in label_filename else 'top'

            label_img = cv2.imread(label_filename, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            if supporting_plane == 'table': # Merge 0 (NaNs) and 1 (non-table/object)
                label_img[label_img == 1] = 0
                label_img = (label_img.astype(int) - 1).clip(min=0).astype(np.uint8)

        elif dataset_name == 'OSD':

            label_img = cv2.imread(label_filename, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            label_img[label_img > 0] = label_img[label_img > 0] + 1 # so values are in [0, 2, 3, ...] (e.g. no table label)

        # Load the prediction (reverse the process I used to go from pcd_files.txt to label_files.txt)
        if dataset_name == 'OCID':
            pred_filename = label_filename.replace(gt_dir, prediction_dir)
            pred_filename = pred_filename.replace('/label/', '/pcd/')
        elif dataset_name == 'OSD':
            pred_filename = label_filename.replace(gt_dir, prediction_dir)
            pred_filename = pred_filename.replace('/OSD-0.2-depth/', '/OSD-0.2/')
            pred_filename = pred_filename.replace('/annotation/', '/pcd/')
        pred_img = util_.imread_indexed(pred_filename)

        # Compare them
        metrics_dict = multilabel_metrics(pred_img, label_img)
        metrics[label_filename] = metrics_dict

        if dataset_name == 'OCID':
            OCID_subset_metrics[subset][label_filename] = metrics_dict
            OCID_subset_metrics[subset + '_' + supporting_plane][label_filename] = metrics_dict
            OCID_subset_metrics[subset + '_' + supporting_plane + '_' + height][label_filename] = metrics_dict

        # Debugging
        # print(label_filename)
        # print(f'Overlap  F: {metrics_dict["Objects F-measure"]}')
        # print(f'Boundary F: {metrics_dict["Boundary F-measure"]}')

    # Compute mean of all metrics
    obj_F_mean = np.mean([metrics[key]['Objects F-measure'] for key in metrics.keys()])
    obj_P_mean = np.mean([metrics[key]['Objects Precision'] for key in metrics.keys()])
    obj_R_mean = np.mean([metrics[key]['Objects Recall'] for key in metrics.keys()])
    boundary_F_mean = np.mean([metrics[key]['Boundary F-measure'] for key in metrics.keys()])
    boundary_P_mean = np.mean([metrics[key]['Boundary Precision'] for key in metrics.keys()])
    boundary_R_mean = np.mean([metrics[key]['Boundary Recall'] for key in metrics.keys()])
    obj_det_075_percentage_mean = np.mean([metrics[key]['obj_detected_075_percentage'] for key in metrics.keys()])

    ret_dict = {
        'obj_F_mean' : obj_F_mean,
        'obj_P_mean' : obj_P_mean,
        'obj_R_mean' : obj_R_mean,
        'boundary_F_mean' : boundary_F_mean,
        'boundary_P_mean' : boundary_P_mean,
        'boundary_R_mean' : boundary_R_mean,
        'obj_det_075_percentage_mean' : obj_det_075_percentage_mean,
    }


    if dataset_name == 'OCID':

        # Get every subset
        for subset in OCID_subsets:

            mdict = OCID_subset_metrics[subset]
            obj_F_mean = np.mean([mdict[key]['Objects F-measure'] for key in mdict.keys()])
            obj_P_mean = np.mean([mdict[key]['Objects Precision'] for key in mdict.keys()])
            obj_R_mean = np.mean([mdict[key]['Objects Recall'] for key in mdict.keys()])
            boundary_F_mean = np.mean([mdict[key]['Boundary F-measure'] for key in mdict.keys()])
            boundary_P_mean = np.mean([mdict[key]['Boundary Precision'] for key in mdict.keys()])
            boundary_R_mean = np.mean([mdict[key]['Boundary Recall'] for key in mdict.keys()])
            obj_det_075_percentage_mean = np.mean([mdict[key]['obj_detected_075_percentage'] for key in mdict.keys()])

            ret_dict[subset] = {
                'obj_F_mean' : obj_F_mean,
                'obj_P_mean' : obj_P_mean,
                'obj_R_mean' : obj_R_mean,
                'boundary_F_mean' : boundary_F_mean,
                'boundary_P_mean' : boundary_P_mean,
                'boundary_R_mean' : boundary_R_mean,
                'obj_det_075_percentage_mean' : obj_det_075_percentage_mean,
            }

    return ret_dict
