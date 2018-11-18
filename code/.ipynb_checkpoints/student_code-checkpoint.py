import numpy as np
import cyvlfeat as vlfeat
from utils import *
import os.path as osp
from glob import glob
from random import shuffle
from IPython.core.debugger import set_trace
from sklearn.svm import LinearSVC

import cv2

def get_positive_features(train_path_pos, feature_params):
    """
    This function should return all positive training examples (faces) from
    36x36 images in 'train_path_pos'. Each face should be converted into a
    HoG template according to 'feature_params'.

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features

    Args:
    -   train_path_pos: (string) This directory contains 36x36 face images
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slower because the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time
            (although you don't have to make the detector step size equal a
            single HoG cell).

    Returns:
    -   feats: N x D matrix where N is the number of faces and D is the template
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    positive_files = glob(osp.join(train_path_pos, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################
    
    n_cell = np.ceil(win_size/cell_size).astype('int')
    feats = np.random.rand(len(positive_files), n_cell*n_cell*31)

    print(feats.shape,win_size,cell_size,len(positive_files),n_cell)
    i = 0
    for file in positive_files:
        im = load_image_gray(file)
        # print(im.shape)
        hog_feats = vlfeat.hog.hog(im, cell_size)
        feats[i,:] = np.reshape(hog_feats,(1, ((win_size//cell_size)**2)*31))
        i = i+1



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    """
    This function should return negative training examples (non-faces) from any
    images in 'non_face_scn_path'. Images should be loaded in grayscale because
    the positive training data is only available in grayscale (use
    load_image_gray()).

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   num_samples: number of negatives to be mined. It is not important for
            the function to find exactly 'num_samples' non-face features. For
            example, you might try to sample some number from each image, but
            some images might be too small to find enough.

    Returns:
    -   N x D matrix where N is the number of non-faces and D is the feature
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    n_cell = np.ceil(win_size/cell_size).astype('int')
    feats = np.random.rand(len(negative_files*6*6), n_cell*n_cell*31)

    print(feats.shape,win_size,cell_size,len(negative_files),n_cell)

    k = 0
    for file in negative_files:
        im = load_image_gray(file)
        # print(im.shape,int(im.shape[0]/6),int(im.shape[1]/6))
        for i in range(int((im.shape[1]/6))):
            for j in range(int((im.shape[0]/6))):
                if (i)*int(im.shape[0]/6)+36 < im.shape[0] and (j)*int(im.shape[1]/6)+36 < im.shape[1]:
                    # print(i*int(im.shape[0]/6),(i)*int(im.shape[0]/6)+36,j*int(im.shape[1]/6),(j)*int(im.shape[1]/6)+36)
                    img = im[i*int(im.shape[0]/6):(i)*int(im.shape[0]/6)+36,j*int(im.shape[1]/6):(j)*int(im.shape[1]/6)+36] #cv2.resize(im,(36,36)) # change
                    # print(img.shape)
                    # print(36-img.shape[0], 36-img.shape[1])
                    # img = np.pad(img, (36-img.shape[1], 36-img.shape[0]), 'edge')
                    # print(img.shape)
                    # img = cv2.resize(img,(36,36),interpolation = cv2.INTER_AREA)
                    hog_feats = vlfeat.hog.hog(img, cell_size)
                    # print(hog_feats.shape)
                    feats[k,:] = np.reshape(hog_feats,(1, ((win_size//cell_size)**2)*31))
                    k = k+1
    feats = feats[:k,:]
    print(k,feats.shape)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats

def train_classifier(features_pos, features_neg, C):
    """
    This function trains a linear SVM classifier on the positive and negative
    features obtained from the previous steps. We fit a model to the features
    and return the svm object.

    Args:
    -   features_pos: N X D array. This contains an array of positive features
            extracted from get_positive_feats().
    -   features_neg: M X D array. This contains an array of negative features
            extracted from get_negative_feats().

    Returns:
    -   svm: LinearSVC object. This returns a SVM classifier object trained
            on the positive and negative features.
    """
    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    # svm = PseudoSVM(10,features_pos.shape[1])
    c_ = 1e-3
    clf = LinearSVC(C = c_)
    y_pos =  np.ones(((features_pos.shape[0]),1))
    y_neg = np.ones(((features_neg.shape[0],1)))*-1
    y_train = np.vstack((y_pos,y_neg))
    X_train = np.vstack((features_pos,features_neg))
    svm = clf.fit(X_train,y_train)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return svm

def mine_hard_negs(non_face_scn_path, svm, feature_params):
    """
    This function is pretty similar to get_random_negative_features(). The only
    difference is that instead of returning all the extracted features, you only
    return the features with false-positive prediction.

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features
    -   svm.predict(feat): predict features

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   svm: LinearSVC object

    Returns:
    -   N x D matrix where N is the number of non-faces which are
            false-positive and D is the feature dimensionality.
    """

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    n_cell = np.ceil(win_size/cell_size).astype('int')
    feats = np.random.rand(len(negative_files), n_cell*n_cell*31)
    # print(len(negative_files),len(feature_params))
    # # y_test = np.ones((feature_params.shape[0]))*-1
    # y_pred = svm.predict(feature_params)
    # print(y_pred)

    i = 0
    for file in negative_files:
        im = load_image_gray(file)
        im = cv2.resize(im, (36, 36))  # change
        # print(im.shape)
        hog_feats = vlfeat.hog.hog(im, cell_size)
        # print(hog_feats.shape)

        features = np.reshape(hog_feats, (1, ((win_size // cell_size) ** 2) * 31))
        y_pred = svm.predict(features)
        # print(y_pred)
        if(y_pred == 1):
            feats[i,:] = features
            i = i + 1

    # print(i)
    feats = feats[:i-1,:]
    # print(feats)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats

def run_detector(test_scn_path, svm, feature_params, verbose=False):
    """
    This function returns detections on all of the images in a given path. You
    will want to use non-maximum suppression on your detections or your
    performance will be poor (the evaluation counts a duplicate detection as
    wrong). The non-maximum suppression is done on a per-image basis. The
    starter code includes a call to a provided non-max suppression function.

    The placeholder version of this code will return random bounding boxes in
    each test image. It will even do non-maximum suppression on the random
    bounding boxes to give you an example of how to call the function.

    Your actual code should convert each test image to HoG feature space with
    a _single_ call to vlfeat.hog.hog() for each scale. Then step over the HoG
    cells, taking groups of cells that are the same size as your learned
    template, and classifying them. If the classification is above some
    confidence, keep the detection and then pass all the detections for an
    image to non-maximum suppression. For your initial debugging, you can
    operate only at a single scale and you can skip calling non-maximum
    suppression. Err on the side of having a low confidence threshold (even
    less than zero) to achieve high enough recall.

    Args:
    -   test_scn_path: (string) This directory contains images which may or
            may not have faces in them. This function should work for the
            MIT+CMU test set but also for any other images (e.g. class photos).
    -   svm: A trained sklearn.svm.LinearSVC object
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slowerbecause the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time.
    -   verbose: prints out debug information if True

    Returns:
    -   bboxes: N x 4 numpy array. N is the number of detections.
            bboxes(i,:) is [x_min, y_min, x_max, y_max] for detection i.
    -   confidences: (N, ) size numpy array. confidences(i) is the real-valued
            confidence of detection i.
    -   image_ids: List with N elements. image_ids[i] is the image file name
            for detection i. (not the full path, just 'albert.jpg')
    """
    im_filenames = sorted(glob(osp.join(test_scn_path, '*.jpg')))
    bboxes = np.empty((0, 4))
    confidences = np.empty(0)
    image_ids = []

    # number of top detections to feed to NMS
    topk = 40 #15

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    # scale_factor = feature_params.get('scale_factor', 0.65)
    template_size = int(win_size / cell_size)


    for idx, im_filename in enumerate(im_filenames):
        print('Detecting faces in {:s}'.format(im_filename))
        im = load_image_gray(im_filename)
        im_id = osp.split(im_filename)[-1]
        im_shape = im.shape
        # print(im_shape)
        # create scale space HOG pyramid and return scores for prediction

        #######################################################################
        #                        TODO: YOUR CODE HERE                         #
        #######################################################################

        cur_x_min = np.empty((0,1))
        cur_y_min = np.empty((0, 1))
        cur_bboxes = np.empty((0, 4))
        cur_confidences = np.empty((0))
        # image_ids = []
        k = 0

        conf = np.ones((1,))

        conf_threshold = -0.8  # confidence threshold
        stride = 2
        p = 0
        scale_factor_list = [1,0.8,0.65]

        for scale_factor in scale_factor_list:
            print(scale_factor)
            # print(im.shape,im_shape)
            img = cv2.resize(im, (int(scale_factor*im_shape[1]),int(scale_factor*im_shape[0])))
            # print(im.shape)
            features = vlfeat.hog.hog(img, cell_size)
            # print(features.shape)

            for i in range(features.shape[0]):
                for j in range(features.shape[1]):
                    x_min = stride*i
                    y_min = stride*j
                    if x_min+template_size < features.shape[0] and y_min+template_size < features.shape[1]:
                        feat_window = features[x_min:x_min+template_size,y_min:y_min+template_size]
                        feat_window_reshaped = np.reshape(feat_window,(1, ((win_size // cell_size) ** 2) * 31))

                        conf = svm.decision_function(feat_window_reshaped)

                        if conf >= conf_threshold:
                            k = k + 1
                            # print(x_min,x_min*template_size/scale_factor,template_size/scale_factor,y_min,y_min*template_size/scale_factor)
                            cur_x_min = np.vstack((cur_x_min,x_min))
                            cur_y_min = np.vstack((cur_y_min,y_min))

                            cur_confidences = np.hstack((cur_confidences,conf))

                            cur_bboxes = np.hstack([cur_y_min*template_size/scale_factor, cur_x_min*template_size/scale_factor, \
                                                    ((cur_y_min + template_size)*template_size/scale_factor).astype('int'), \
                                                    ((cur_x_min + template_size)*template_size/scale_factor).astype('int')])

                            print(i, j, cur_bboxes)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

        ### non-maximum suppression ###
        # non_max_supr_bbox() can actually get somewhat slow with thousands of
        # initial detections. You could pre-filter the detections by confidence,
        # e.g. a detection with confidence -1.1 will probably never be
        # meaningful. You probably _don't_ want to threshold at 0.0, though. You
        # can get higher recall with a lower threshold. You should not modify
        # anything in non_max_supr_bbox(). If you want to try your own NMS methods,
        # please create another function.

        idsort = np.argsort(-cur_confidences)[:topk]
        cur_bboxes = cur_bboxes[idsort]
        cur_confidences = cur_confidences[idsort]

        is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences,
            im_shape, verbose=verbose)

        if(len(is_valid_bbox)):
            print('NMS done, {:d} detections passed'.format(sum(is_valid_bbox)))
            cur_bboxes = cur_bboxes[is_valid_bbox]
            cur_confidences = cur_confidences[is_valid_bbox]

            bboxes = np.vstack((bboxes, cur_bboxes))
        confidences = np.hstack((confidences, cur_confidences))
        image_ids.extend([im_id] * len(cur_confidences))

    # print(bboxes.shape,confidences.shape,len(image_ids),k)

    return bboxes, confidences, image_ids
