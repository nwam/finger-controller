from vision import MHB

class RunProcessor:
    def __init__(self, cnn_input, game_input):
        self.mhb = MHB(cnn_input, 90)
        self.game_input = game_input
        self.h_pos_ratio = 0.425
        self.h_speed_alpha = 0.2
        self.h_speed_thresh = 5.0
        self.h_speed = self.h_speed_thresh
        self.h_pos_alpha = 0.3
        self.h_pos_thresh = self.mhb.hmag.shape[1] * self.h_pos_ratio
        self.h_pos = self.h_pos_thresh
        self.h_classes = ['run', 'walk', 'movef', 'moveb']

    def process(self, class_label):
        if class_label not in self.h_classes:
            self.h_speed = self.h_speed_thresh
            self.h_pos = self.h_pos_thresh
            return class_label
        else:
            mhb_speed, mhb_pos = self.mhb.compute()

            self.h_pos = self.h_pos_alpha*mhb_pos[1] + (1-self.h_pos_alpha)*self.h_pos
            if self.h_pos > self.h_pos_thresh:
                self.game_input.direction_forward()
            else:
                self.game_input.direction_backward()

            self.h_speed = self.h_speed_alpha*mhb_speed + (1-self.h_speed_alpha)*self.h_speed

            if class_label in ['run', 'walk']:
                if self.h_speed > self.h_speed_thresh:
                    return 'run'
                else:
                    return 'walk'
            else:
                return class_label

class StickyTolerance:
    def __init__(self):
        self.prev_label = None
        self.tolerance = 0.7
        self.tolerance_patience = 2
        self.patience = self.tolerance_patience
        self.sticky_size = 2
        self.sticky = 0
        self.sticky_i = 0

    def process(self, class_label, certainty, action):
        if certainty < self.tolerance:
            self.patience -= 1
            if self.patience < 0:
                action = None
            else:
                action = self.prev_label
        else:
            self.patience = self.tolerance_patience

        if class_label != self.prev_label:
            self.sticky_i = 0
        if self.sticky_i < self.sticky_size:
            self.sticky_i += 1
        else:
            action = class_label
        self.prev_label = class_label

        return action
