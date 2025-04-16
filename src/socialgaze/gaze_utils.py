

class GazePositions:
    def __init__(self, x, y):
        self._x_pos = x
        self._y_pos = y

    def get_x_pos(self):
        return self._x_pos
    
    def get_y_pos(self):
        return self._y_pos
    


class GazeData:

    def __init__(self):
        self._positions = []
        self._pupil = []
        self._roi_vertices = None
        self._neural_timeline = None

    def add_gaze_pos_list(self, input_list):
        for x, y in input_list:
            gaze_pos = GazePositions(x, y)
            self._positions.append(gaze_pos)


    