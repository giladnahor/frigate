import frigate.detection_pb2 as detection_pb2
from google.protobuf.json_format import MessageToJson
import numpy as np


def add_image(frame: detection_pb2.Frame, img):
    shape = img.shape
    frame.data = img.tobytes()
    frame.type = frame.RGB
    frame.Width = shape[0]
    frame.Height = shape[1]
    if len(shape) > 2:
        frame.Channels = shape[2]
    else:
        frame.Channels = 0


def get_image(frame: detection_pb2.Frame):
    img = np.frombuffer(frame.data, dtype=np.uint8)
    img = img.reshape(frame.Height, frame.Width, frame.Channels)
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
        det.label = "person"
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
