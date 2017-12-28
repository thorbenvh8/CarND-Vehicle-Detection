import numpy as np

# Define a class to receive the characteristics of each line detection
class Car():
    def __init__(self):
        # x values of the last found centers for the car
        self.recent_center = []
        #average x values of the fitted line over the last n iterations
        self.avg_center = None

        self.recent_margin_h = []

        self.avg_window = None
        #n iterations
        self.n = 10

        self.fit_margin = 60

        self.found_count = 0

        self.not_found_count = 0

    def add_center(self, center):
        self.found_count += 1
        self.not_found_count = 0
        self.recent_center.append(center)
        if len(self.recent_center) > self.n:
            self.recent_center.pop(0)
        self.avg_center = np.average(self.recent_center, axis=0)
        center_h = int(self.avg_center[0])
        center_v = int(self.avg_center[1])
        margin_h = (420 - center_v) * 3
        self.recent_margin_h.append(margin_h)
        if len(self.recent_margin_h) > self.n:
            self.recent_margin_h.pop(0)
        self.avg_margin_h = int(np.average(self.recent_margin_h))
        margin_v = int(self.avg_margin_h * 3/4)
        self.avg_window = ((center_h-self.avg_margin_h, center_v-margin_v), (center_h+self.avg_margin_h, center_v+margin_v))

    def is_fit(self, center):
        return self.avg_center[0] - self.fit_margin <= center[0] and self.avg_center[0] + self.fit_margin >= center[0] and self.avg_center[1] - self.fit_margin <= center[1] and self.avg_center[1] + self.fit_margin >= center[1]

    def enough_found():
        return self.found_count >= 3
