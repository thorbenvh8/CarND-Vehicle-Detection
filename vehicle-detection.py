from save_load_file import load_model, load_frames, save_frames
import lessons
import cv2
import matplotlib.pyplot as plt
import util
import features
from scipy.ndimage.measurements import label
import numpy as np
from car import Car

COUNT_FRAMES_DELETE_CARS = 5
COUNT_FRAMES_DRAW_CAR = 10

COLORS=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]
VIDEO_FILENAME = "test_video"
VIDEO_FILENAME = "project_video"
cars = []

def generate_windows_list(n, size_start, size_end, overlap_start, overlap_end, y_start, y_end):
    print("Generating windows list ...")
    windows_list = []
    step_size = (size_end - size_start) / (n-1)
    step_overlap = (overlap_end - overlap_start) / (n-1)
    step_y = (y_end - y_start) / (n-1)
    for i in range(n):
        size = int(size_start + step_size * i)
        overlap = overlap_start + step_overlap * i
        y = int(y_start + step_y * i)
        windows = lessons.slide_window(frames[0], x_start_stop=[None, None], y_start_stop=[y, y + size],
                            xy_window=(size, size), xy_overlap=(overlap, 0.5))
        windows_list.append(windows)
    print("Generated windows list")
    return windows_list

def range_overlap(a_min, a_max, b_min, b_max):
    '''Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)

def sort_windows(window):
    return window[0][0]

def get_heat_windows(windows_pixels, x_indices, y_indices):
    windows = []

    for i in range(len(x_indices)):
        x1 = x_indices[i]
        y1 = y_indices[i]
        exists = False
        for window in windows:
            w_y1 = window[0][0]
            w_x1 = window[0][1]
            w_y2 = window[1][0]
            w_x2 = window[1][1]
            if w_y1 <= y1 and y1 <= w_y2 and w_x1 <= x1 and x1 <= w_x2:
                exists = True
                break
        if not exists:
            x2, y2 = get_heat_windows_rec(windows_pixels, x1, y1)
            #((x1, y1), (x2, y2))
            if x2 - x1 > 15 and y2 - y1 > 15:
                windows.append(((y1,x1),(y2,x2)))

    sorted(windows, key=sort_windows)

    margin = 15
    i = 0
    check_again = False
    while i < len(windows):
        w1 = windows[i]
        w1_top = w1[0][0]
        w1_left = w1[0][1]
        w1_right = w1[1][1]
        w1_bottom = w1[1][0]
        j = 0
        while j < len(windows):
            if i == j:
                j += 1
                continue
            w2 = windows[j]
            w2_top = w2[0][0]
            w2_left = w2[0][1]
            w2_bottom = w2[1][0]
            w2_right = w2[1][1]
            if w1_left <= w2_left and w1_right + margin >= w2_left and range_overlap(w2_top, w2_bottom, w1_top, w1_bottom):
                windows[i] = ((min(w1_top, w2_top), min(w1_left, w2_left)), (max(w1_bottom, w2_bottom), max(w1_right, w2_right)))
                del windows[j]
                check_again = True
                break
            if w1_top <= w2_top and w1_bottom + margin >= w2_top and range_overlap(w2_left, w2_right, w1_left, w1_right):
                windows[i] = ((min(w1_top, w2_top), min(w1_left, w2_left)), (max(w1_bottom, w2_bottom), max(w1_right, w2_right)))
                del windows[j]
                check_again = True
                break
            j += 1
        if check_again:
            check_again = False
            continue
        i += 1
    return windows

def get_heat_windows_rec(windows_pixels, x, y):
    x_resultx = x
    y_resultx = y
    x_resulty = x
    y_resulty = y
    if windows_pixels[x+1][y+1] == 1:
        x_resultx, y_resultx = get_heat_windows_rec(windows_pixels, x+1, y+1)
    elif windows_pixels[x+2][y+2] == 1:
        x_resultx, y_resultx = get_heat_windows_rec(windows_pixels, x+2, y+2)
    elif windows_pixels[x+3][y+3] == 1:
        x_resultx, y_resultx = get_heat_windows_rec(windows_pixels, x+3, y+3)
    #if windows_pixels[x][y+1] == 1:
    #    x_resulty, y_resulty = get_windows_rec(windows_pixels, x, y+1)
    return x_resultx, y_resultx

def generate_heatmap(frame, windows, threshold):
    heatmap = np.zeros_like(frame)
    # Iterate through all windows
    for window in windows:
        # Increase number of windows containing pixels
        # Assuming each window takes the form ((x1, y1), (x2, y2))
        heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0], 0] += 1

    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0

    # Transform into red color spectrum from 0 - 255
    max_threshold = 3
    min_count = 3
    max_heatmap = max(np.amax(heatmap) - max_threshold, 3)
    x_indices,y_indices,z_indices = np.nonzero(heatmap >=max_heatmap)

    windows_pixels = np.zeros_like(frame[:,:,0])
    #for index in indizes:
    for i in range(len(x_indices)):
        windows_pixels[x_indices[i],y_indices[i]] = 1
        heatmap[x_indices[i],y_indices[i]] = [0,max_heatmap,0]

    heat_windows = get_heat_windows(windows_pixels, x_indices, y_indices)
    vehicle_windows = get_vehicle_windows(heatmap, heat_windows)



    #heatmap[heatmap == [max_heatmap,0,0]] = [max_heatmap,max_heatmap,max_heatmap]
    heatmap = heatmap / max_heatmap * 255

    return np.array(heatmap, dtype=np.uint8), heat_windows

def detect_vehicles(svc, frames, windows_list, orient, pix_per_cell, cell_per_block, cspace, spatial_size, hist_bins, hist_range):
    enhanced_frames = []
    len_frames = len(frames)
    util.printProgressBar(0, len_frames, 'Detecting vehicles')
    for i in range(len_frames):
        frame = frames[i]
        vehicle_windows = []
        for j in range(len(windows_list)):
            for k in range(len(windows_list[j])):
                window = windows_list[j][k]
                #(startx, starty), (endx, endy)
                startx = window[0][0]
                starty = window[0][1]
                endx = window[1][0]
                endy = window[1][1]
                diffx = endx - startx
                diffy = endy - starty
                diffxy = min(diffx, diffy)
                window_frame = frame[starty:starty+diffxy,startx:startx+diffxy]
                window_frame_resize = cv2.resize(window_frame, (64,64))
                window_frame_resize_features = features.get_features(window_frame_resize, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, cspace=cspace, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range)
                predictions = svc.predict([window_frame_resize_features])
                for prediction in predictions:
                    if prediction == 1:
                        vehicle_windows.append(window)

        threshold = 1
        heatmap, heat_windows = generate_heatmap(frame, vehicle_windows, threshold)

        for car in cars:
            car.not_found_count += 1

        veh_windows = []
        for heat_window in heat_windows:
            left = heat_window[0][0]
            top = heat_window[0][1]
            right = heat_window[1][0]
            bottom = heat_window[1][1]
            center_h = left + int((right-left)/2)
            center_v = top + int((bottom-top)/2)
            center = (center_h, center_v)
            car_exists = False
            for car in cars:
                if car.is_fit(center):
                    car.add_center(center)
                    if car.found_count >= COUNT_FRAMES_DRAW_CAR:
                        veh_windows.append(car.avg_window)
                    car_exists = True
                    break

            if not car_exists:
                car = Car()
                cars.append(car)
                car.add_center(center)

        for car in cars:
            if car.not_found_count >= COUNT_FRAMES_DELETE_CARS:
                cars.remove(car)

        frame = lessons.draw_boxes(frame, veh_windows, color=(0, 0, 255), thick=3)

        enhanced_frames.append(frame)
        util.printProgressBar(i+1, len_frames, 'Detecting vehicles')
    return enhanced_frames

svc, orient, pix_per_cell, cell_per_block, cspace, spatial_size, hist_bins, hist_range = load_model()
frames, fps = load_frames(VIDEO_FILENAME + ".mp4")#, start_frame=25*23, end_frame=25*25)

windows_list = generate_windows_list(n=10, size_start=16, size_end=213, overlap_start=0.5, overlap_end=0.8, y_start=416, y_end=380)
enhanced_frames = detect_vehicles(svc, frames, windows_list, orient, pix_per_cell, cell_per_block, cspace, spatial_size, hist_bins, hist_range)

save_frames(enhanced_frames, fps, VIDEO_FILENAME + "_output.mp4")
