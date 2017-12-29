import util
import cv2
import matplotlib.image as mpimg
import lessons
import numpy as np

def get_features(img, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), hog_channel='ALL'):
    color_features = lessons.extract_features(img, cspace=cspace, spatial_size=spatial_size,
                            hist_bins=hist_bins, hist_range=hist_range)

    '''if hog_channel == 'ALL':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = img[:,:,hog_channel]'''

    if vis == True:
        hog_features, img_hog = lessons.get_hog_features(img, orient=orient,
                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis)
        return np.concatenate((hog_features, color_features), axis=0), img_hog
    else:
        hog_features_c1 = lessons.get_hog_features(img[:,:,1], orient=orient,
                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis)
        hog_features_c2 = lessons.get_hog_features(img[:,:,2], orient=orient,
                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis)
        #return hog_features
        #hog_features = np.array(hog_features / 255, dtype=np.float64)
        return np.concatenate((hog_features_c1, hog_features_c2, color_features), axis=0)


def get_features_list(name, img_paths, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), hog_channel='ALL'):
    features = []
    img_hogs = []
    len_img_paths = len(img_paths)
    util.printProgressBar(0, len_img_paths, name)
    for i in range(len_img_paths):
        img_path = img_paths[i]
        img = mpimg.imread(img_path)
        if vis == True:
            img_features, img_hog = get_features(img, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis, cspace=cspace, spatial_size=spatial_size,
                                    hist_bins=hist_bins, hist_range=hist_range, hog_channel=hog_channel)
            img_hogs.append(img_hog)
        else:
            img_features = get_features(img, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis, cspace=cspace, spatial_size=spatial_size,
                                    hist_bins=hist_bins, hist_range=hist_range, hog_channel=hog_channel)
        features.append(img_features)
        util.printProgressBar(i+1, len_img_paths, name)
    if vis == True:
        return features, img_hogs

    return features
