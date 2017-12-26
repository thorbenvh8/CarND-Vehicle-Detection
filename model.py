import lessons
import util
import glob
from sklearn.model_selection import train_test_split
import cv2
from sklearn.svm import LinearSVC
import time
from save_to_file import save_model

TEST_SIZE=0.2
ORIENT=9
PIX_PER_CELL=8
CELL_PER_BLOCK=2

def load_data(orient, pix_per_cell, cell_per_block, test_size=0.2):
    print("Test size: {:0.2f}%".format(test_size*100))

    vehicles = glob.glob('./vehicles/*/*.png')
    non_vehicles = glob.glob('./non-vehicles/*/*.png')

    vehicles_train, vehicles_test = train_test_split(vehicles, test_size=test_size)
    print("Vehicles: {}".format(len(vehicles)))
    print("Vehicles train: {}".format(len(vehicles_train)))
    print("Vehicles test:  {}".format(len(vehicles_test)))

    non_vehicles_train, non_vehicles_test = train_test_split(non_vehicles, test_size=test_size)
    print("Non vehicles: {}".format(len(non_vehicles)))
    print("Non vehicles train: {}".format(len(non_vehicles_train)))
    print("Non vehicles test:  {}".format(len(non_vehicles_test)))

    X_train = vehicles_train + non_vehicles_train
    X_train = get_features("Load features X_train", X_train, orient=ORIENT, pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK)
    y_train = [1] * len(vehicles_train) + [0] * len(non_vehicles_train)
    X_train, y_train = util.shuffle_list(X_train, y_train)

    X_test = vehicles_test + non_vehicles_test
    X_test = get_features("Load features X_test", X_test, orient=ORIENT, pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK)
    y_test = [1] * len(vehicles_test) + [0] * len(non_vehicles_test)
    X_test, y_test = util.shuffle_list(X_test, y_test)

    return X_train, y_train, X_test, y_test

def get_features(name, img_paths, orient=9, pix_per_cell=8, cell_per_block=2, vis=False):
    features = []
    img_hogs = []
    len_img_paths = len(img_paths)
    util.printProgressBar(0, len_img_paths-1, name)
    for i in range(len_img_paths):
        img_path = img_paths[i]
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if vis == True:
            img_features, img_hog = lessons.get_hog_features(gray, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis)
            img_hogs.append(img_hog)
        else:
            img_features = lessons.get_hog_features(gray, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=vis)
        features.append(img_features)

        util.printProgressBar(i, len_img_paths-1, name)
    if vis == True:
        return features, img_hogs
    return features

def train_model(X_train, y_train):
    # Use a linear SVC (support vector classifier)
    svc = LinearSVC()
    # Train the SVC
    print("Training ...")
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to train')
    return svc

def test_model(svc, X_test, y_test):
    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

X_train, y_train, X_test, y_test = load_data(orient=ORIENT, pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, test_size=TEST_SIZE)
svc = train_model(X_train, y_train)
test_model(svc, X_test, y_test)
save_model(svc, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK)
