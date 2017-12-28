import util
import cv2
import lessons

def get_features(img, orient=9, pix_per_cell=8, cell_per_block=2, gray_convert=cv2.COLOR_BGR2GRAY, vis=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if vis == True:
        img_features, img_hog = lessons.get_hog_features(gray, orient=orient,
                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis)
        return img_features, img_hog
    else:
        img_features = lessons.get_hog_features(gray, orient=orient,
                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis)
        return img_features


def get_features_list(name, img_paths, orient=9, pix_per_cell=8, cell_per_block=2, gray_convert=cv2.COLOR_BGR2GRAY, vis=False):
    features = []
    img_hogs = []
    len_img_paths = len(img_paths)
    util.printProgressBar(0, len_img_paths, name)
    for i in range(len_img_paths):
        img_path = img_paths[i]
        img = cv2.imread(img_path)
        if vis == True:
            img_features, img_hog = get_features(img, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, gray_convert=gray_convert, vis=vis)
            img_hogs.append(img_hog)
        else:
            img_features = get_features(img, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, gray_convert=gray_convert, vis=vis)
        features.append(img_features)
        util.printProgressBar(i+1, len_img_paths, name)
    if vis == True:
        return features, img_hogs
    return features
