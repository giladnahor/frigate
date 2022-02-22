import cv2
import time

import frigate.detection_pb2 as detection_pb2
from google.protobuf.json_format import MessageToJson
import numpy as np


def add_image(frame: detection_pb2.Frame, img):
    shape = img.shape
    frame.data = img.tobytes()
    frame.type = frame.RGB
    frame.Width = shape[1]
    frame.Height = shape[0]


def get_image(frame: detection_pb2.Frame, outputI420=True):
    img = np.frombuffer(frame.data, dtype=np.uint8)
    if frame.type == frame.FrameType.RGB:
        img = img.reshape(frame.Height, frame.Width, 3)
        if outputI420:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420)
    if frame.type == frame.FrameType.YUY2:
        img = img.reshape(frame.Height, frame.Width, 2)
        if outputI420:
            assert "outputI420 not supported"
    if frame.type == frame.FrameType.I420:
        img = img.reshape(int(frame.Height * 1.5), frame.Width)
    return img


def example():
    # Create new frame
    frame = detection_pb2.Frame()

    img = np.zeros((640, 480, 3), dtype=np.uint8)
    img[0, :, :] = 1
    add_image(frame, img)

    for i in range(2, 6):
        det = frame.Detections.add()
        det.xmin = i
        det.xmax = i
        det.ymin = i
        det.ymax = i
        det.det_class = i
        det.score = i

    # serialize

    json_msg = MessageToJson(frame)
    print(f"JSON size {len(json_msg)}")

    buf = frame.SerializeToString()
    print(f"BUF size {len(buf)}")

    # de-serialize
    new_frame = detection_pb2.Frame().FromString(buf)
    for detection in new_frame.Detections:
        print(detection)
    new_img = get_image(new_frame)
    print(new_img.shape)


class fps_measure(object):
    def __init__(
        self, control, alpha=0.15, avg_period=1, verbose=False, name="default"
    ):
        self._name = name
        self._verbose = verbose
        self._avg = 0
        self._alpha = alpha
        self._last_time = None
        self._count = 1
        self._avg_period = avg_period
        self._control = control
        self._fps_measurements = None
        self._initial_sample = True
        fps_measurement_name = "FPS_{}".format(self._name)
        if self._control is not None:  # To use w/o control integration pass None
            if self._control.has_data_collection(fps_measurement_name):
                self._fps_measurements = self._control.get_data_collection(
                    fps_measurement_name
                )

        self._last_time_fps_reported = None

    def get_alpha(self):
        return self._alpha

    def set_alpha(self, alpha):
        self._alpha = alpha

    def get(self):
        return self._avg

    def update(self):
        if self._last_time is None:
            self._last_time = time.time()
            return 0
        cur_time = time.time()
        diff = cur_time - self._last_time
        if diff >= self._avg_period:
            fps = self._count / diff
            if (
                self._initial_sample
            ):  # If this is the first time around, then the fps is the average
                self._avg = fps
                self._initial_sample = False
            fps_error = fps - self._avg
            #  If error is very big compared to average update average directly
            if abs(fps_error / (self._avg + 0.1)) > 0.2:
                self._avg = fps
            else:
                self._avg = self._avg + (self._alpha * fps_error)
            self._last_time = cur_time
            self._count = 1
            if self._verbose:
                print(
                    "FPS measured on {} : avg {} cur {}".format(
                        self._name, self._avg, fps
                    )
                )
        else:
            self._count += 1

        if self._fps_measurements is not None:
            self._report_fps_to_measurements_mechanism()

        return self._avg

    # Used by gstreamer apps to report average fps
    def report_average(self, average_fps):
        if self._initial_sample:  # Set initial sample flag
            self._initial_sample = False

        self._avg = average_fps

        if self._fps_measurements is not None:
            self._report_fps_to_measurements_mechanism()

    def _report_fps_to_measurements_mechanism(self):
        """This method reports the measured FPS every 1 seconds
        The assumption is that this method is being called more often than 1 sec"""
        if self._last_time_fps_reported is None:
            self._last_time_fps_reported = time.time()
            return

        cur_time = time.time()
        diff = cur_time - self._last_time_fps_reported
        # Check if 1 second is passed since last FPS reported:
        if diff >= 1:
            self._last_time_fps_reported = time.time()
            if not self._initial_sample:
                self._fps_measurements.append(self._avg)

    def pause(self):
        self._last_time = None

    def sample(self):
        # this function returns the FPS since the last update
        # w/o updating _last_time
        # used to sample the current step w/o updating the FPS counter
        if self._last_time is None:
            return 0
        diff = time.time() - self._last_time
        fps = 1 / diff
        return fps
