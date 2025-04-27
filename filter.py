import numpy as np
import random
from collections import deque


def box_to_state(box):
    """Convert a bounding box to a state vector."""
    box.extend([0, 0])  # Add velocity components
    return np.asarray(box, dtype=np.float64).reshape((6, 1)) 


def area(state):
    """Calculate the area of a bounding box from the state."""
    return state[2] * state[3]


class BaseKalmanFilter:
    """Base class for Kalman Filters with shared functionality."""

    def __init__(self, box, confidence, id, freq, discard_time, track_length):
        self.x = box_to_state(box)
        self.x_prev = self.x.copy()
        self.A = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.C = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        self.id = id
        self.R = np.identity(6)  # Process noise
        self.Q = np.identity(4) * (1 - confidence)  # Measurement noise
        self.sigma = np.identity(6) * (1 - confidence)
        self.sigma[4, 4] += 1
        self.sigma[5, 5] += 1
        self.time_count = 0
        self.colour = tuple(random.randint(0, 255) for _ in range(3))
        self.freq = freq
        self.discard_time = discard_time
        self.pts = deque(maxlen=track_length)

    def update(self, box, confidence):
        """Update the state with a new measurement."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def predict(self):
        """Predict the next state."""
        self.x = self.A @ self.x
        self.sigma_bar = self.A @ self.sigma @ self.A.T + self.R
        self.time_count += 1
        self.pts.appendleft((
            int(self.x[0] + self.x[2] / 2),
            int(self.x[1] + self.x[3] / 2)
        ))

    def get_state(self):
        """Return the current state."""
        return self.x[:4]

    def in_bounds(self, width, height, margins):
        """Check if the object is within the bounds of the frame."""
        xc = self.x[0] + self.x[2] / 2
        yc = self.x[1] + self.x[3] / 2
        margin = int(min(width, height) * margins)
        return (-margin <= xc <= width + margin) and (-margin <= yc <= height + margin)

    def lost(self):
        """Check if the object is lost."""
        return self.time_count > self.discard_time


class KalmanFilter1(BaseKalmanFilter):
    """First variant of the Kalman Filter."""

    # def update(self, box, confidence):
    #     self.Q = np.identity(4) * (1 - confidence)
    #     self.z = box_to_state(box)[:4, :]
    #     self.K = (self.sigma_bar @ self.C.T) @ np.linalg.inv(self.C @ self.sigma_bar @ self.C.T + self.Q)
    #     self.x += self.K @ (self.z - self.C @ self.x)
    #     self.sigma = (np.identity(6) - self.K @ self.C) @ self.sigma_bar
    #     self.x[4] = (self.x[0] - self.x_prev[0]) / self.time_count
    #     self.x[5] = (self.x[1] - self.x_prev[1]) / self.time_count
    #     self.time_count = 0
    #     self.x_prev = self.x.copy()
    #     self.pts.appendleft((
    #         int(self.x[0] + self.x[2] / 2),
    #         int(self.x[1] + self.x[3] / 2)
    #     ))

    def update(self, box, confidence):
        
        self.Q = np.identity(4)*(1 - confidence)
        
        self.z = box_to_state(box)[0:4, :]
        self.K = (self.sigma_bar @ self.C.T) @ (np.linalg.inv((self.C @ self.sigma_bar @ self.C.T) + self.Q))
        self.x = self.x + self.K@(self.z - (self.C @ self.x))
        self.sigma = (np.identity(6) - (self.K @ self.C)) @ self.sigma_bar


        self.x[4] = (self.x[0] - self.x_prev[0])/self.time_count
        self.x[5] = (self.x[1] - self.x_prev[1])/self.time_count

        self.time_count = 0

        self.x_prev = self.x

        self.pts.appendleft((int(self.x[0] + int(self.x[2]/2)), int(self.x[1] + int(self.x[3]/2)))) #this is for decreasing line



class KalmanFilter2(BaseKalmanFilter):
    """Second variant of the Kalman Filter."""

    def __init__(self, box, confidence, id, freq, discard_time, track_length):
        super().__init__(box, confidence, id, freq, discard_time, track_length)
        self.first = True

    def update(self, box, confidence):
        self.Q = np.identity(4) * (1 - confidence)
        self.z = box_to_state(box)[:4, :]
        self.K = (self.sigma_bar @ self.C.T) @ np.linalg.inv(self.C @ self.sigma_bar @ self.C.T + self.Q)
        self.x += self.K @ (self.z - self.C @ self.x)
        self.sigma = (np.identity(6) - self.K @ self.C) @ self.sigma_bar
        weighting = 1 if self.first else 0.9
        self.x[4] = weighting * (self.x[0] - self.x_prev[0]) / self.time_count + (1 - weighting) * self.x[4]
        self.x[5] = weighting * (self.x[1] - self.x_prev[1]) / self.time_count + (1 - weighting) * self.x[5]
        self.time_count = 0
        self.x_prev = self.x.copy()
        self.first = False
        self.pts.appendleft((
            int(self.x[0] + self.x[2] / 2),
            int(self.x[1] + self.x[3] / 2)
        ))


class KalmanFilter3(BaseKalmanFilter):
    """Third variant of the Kalman Filter."""

    def __init__(self, box, confidence, id, freq, discard_time, track_length):
        super().__init__(box, confidence, id, freq, discard_time, track_length)
        self.updates_counter = 0
        self.areas = []
        self.x_vel = []
        self.y_vel = []
        self.velocity_scaling = 1
        self.init_iterations = 5

    def update(self, box, confidence):
        self.Q = np.identity(4) * (1 - confidence)
        self.z = box_to_state(box)[:4, :]
        self.K = (self.sigma_bar @ self.C.T) @ np.linalg.inv(self.C @ self.sigma_bar @ self.C.T + self.Q)
        self.x += self.K @ (self.z - self.C @ self.x)
        self.sigma = (np.identity(6) - self.K @ self.C) @ self.sigma_bar

        if self.updates_counter <= self.init_iterations:
            self.x[4] = (self.x[0] - self.x_prev[0]) / self.time_count
            self.x[5] = (self.x[1] - self.x_prev[1]) / self.time_count
            self.x_vel.append(self.x[4])
            self.y_vel.append(self.x[5])
            self.areas.append(area(self.x))
            self.x[4] = sum(self.x_vel) / len(self.x_vel)
            self.x[5] = sum(self.y_vel) / len(self.y_vel)
        else:
            self.x[4] = (self.x[0] - self.x_prev[0]) / self.time_count
            self.x[5] = (self.x[1] - self.x_prev[1]) / self.time_count
            self.x_vel.append(self.x[4])
            self.x_vel.pop(0)
            self.y_vel.append(self.x[5])
            self.y_vel.pop(0)
            self.x[4] = sum(self.x_vel) / len(self.x_vel)
            self.x[5] = sum(self.y_vel) / len(self.y_vel)
            avg_area = sum(self.areas) / self.init_iterations
            self.velocity_scaling = 1 - 0.01 * ((avg_area - area(self.x)) / avg_area)
            self.areas.append(area(self.x))
            self.areas.pop(0)

        self.pts.appendleft((
            int(self.x[0] + self.x[2] / 2),
            int(self.x[1] + self.x[3] / 2)
        ))
        self.time_count = 0
        self.x_prev = self.x.copy()
        self.updates_counter += 1
