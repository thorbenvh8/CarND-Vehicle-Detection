import lessons
import util
import glob
from sklearn.model_selection import train_test_split
import cv2
from sklearn.svm import LinearSVC
import time
from save_load_file import save_model
import features

TEST_SIZE=0.2
ORIENT=15
PIX_PER_CELL=8
CELL_PER_BLOCK=2
SPATIAL_SIZE=(16, 16)
HIST_BINS=32
HIST_RANGE=(100, 200)
COLOR_SPACE='HLS'
HOG_CHANNEL=1

def load_data(orient, pix_per_cell, cell_per_block, test_size=0.2):
    print("Test size: {:0.2f}%".format(test_size*100))

    vehicles = glob.glob('./vehicles/*/*.png')
    non_vehicles = glob.glob('./non-vehicles/*/*.png')

    #vehicles = vehicles[0:2]
    #non_vehicles = non_vehicles[0:2]

    vehicles_train, vehicles_test = train_test_split(vehicles, test_size=test_size)
    print("Vehicles: {}".format(len(vehicles)))
    print("Vehicles train: {}".format(len(vehicles_train)))
    print("Vehicles test:  {}".format(len(vehicles_test)))

    non_vehicles_train, non_vehicles_test = train_test_split(non_vehicles, test_size=test_size)
    print("Non vehicles: {}".format(len(non_vehicles)))
    print("Non vehicles train: {}".format(len(non_vehicles_train)))
    print("Non vehicles test:  {}".format(len(non_vehicles_test)))

    X_train = vehicles_train + non_vehicles_train
    X_train = features.get_features_list("Load features X_train", X_train, orient=ORIENT, pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, cspace=COLOR_SPACE, spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, hist_range=HIST_RANGE, hog_channel=HOG_CHANNEL)
    y_train = [1] * len(vehicles_train) + [0] * len(non_vehicles_train)
    X_train, y_train = util.shuffle_list(X_train, y_train)

    X_test = vehicles_test + non_vehicles_test
    X_test = features.get_features_list("Load features X_test", X_test, orient=ORIENT, pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, cspace=COLOR_SPACE, spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, hist_range=HIST_RANGE, hog_channel=HOG_CHANNEL)
    y_test = [1] * len(vehicles_test) + [0] * len(non_vehicles_test)
    X_test, y_test = util.shuffle_list(X_test, y_test)

    return X_train, y_train, X_test, y_test

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
save_model(svc, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, COLOR_SPACE, SPATIAL_SIZE, HIST_BINS, HIST_RANGE)
