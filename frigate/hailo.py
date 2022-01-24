from socket import timeout
import numpy as np
import cv2
from multiprocessing import Process, get_context, Event
from multiprocessing.queues import Queue

from hailo_platform import (
    HEF,
    PcieDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    InputVStreams,
    OutputVStreams,
    FormatType,
)
from frigate.hailo_postprocess import DetectionPostNMS

import time
import queue


def infer(
    network_group,
    input_vstreams_params,
    output_vstreams_params,
    network_group_params,
    postprocess_cb,
    input_queue,
    output_queue,
    stop_event,
):
    with InferVStreams(
        network_group,
        input_vstreams_params,
        output_vstreams_params,
        tf_nms_format=True,
    ) as infer_pipeline:
        with network_group.activate(network_group_params):
            while not stop_event.is_set():
                try:
                    input_tensor = input_queue.get(timeout=1)
                except queue.Empty:
                    continue
                infer_results = infer_pipeline.infer(input_tensor)
                detections = postprocess_cb(infer_results)
                # print("Detections {}".format(detections))
                output_queue.put(detections)


def send(configured_network, input_queue, bypass_queue, stop_event):
    input_vstreams_params = InputVStreamParams.make(
        configured_network, quantized=False, format_type=FormatType.UINT8
    )
    input_names = configured_network.get_input_vstream_infos()
    configured_network.wait_for_activation(1000)
    with InputVStreams(configured_network, input_vstreams_params) as vstreams:
        input_streams = [vstreams.get(layer.name) for layer in input_names]
        while not stop_event.is_set():
            try:
                (frame, metadata) = input_queue.get(timeout=1)
            except queue.Empty:
                continue
            for stream in input_streams:
                stream.send(frame)
            bypass_queue.put(metadata)
            time.sleep(0)


def recv(configured_network, output_queue, bypass_queue, stop_event):
    output_vstreams_params = OutputVStreamParams.make(
        configured_network, quantized=False, format_type=FormatType.UINT8
    )
    configured_network.wait_for_activation(1000)
    hw_output = {}
    with OutputVStreams(configured_network, output_vstreams_params) as vstreams:
        while not stop_event.is_set():
            try:
                metadata = bypass_queue.get(timeout=1)  # get metadata
            except queue.Empty:
                continue
            try:
                for stream in vstreams:
                    hw_output[stream.info.name] = stream.recv()
            except Exception as e:
                raise e
            output_queue.put((hw_output, metadata))
            time.sleep(0)


class hailo:
    def __init__(self) -> None:
        self.stop_event = Event()
        # The target can be used as a context manager ("with" statement) to ensure it's released on time.
        # Here it's avoided for the sake of simplicity
        self.target = PcieDevice()

        # Loading compiled HEFs to device:
        self.model_name = "ssd_mobilenet_v1"
        hef_path = "/media/frigate/ssd_mobilenet_v1.hef"  # "../hefs/{}.hef".format(self.model_name)
        self.hef = HEF(hef_path)

        # Configure network groups
        self.configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = network_groups[0]
        self.network_group_params = self.network_group.create_params()

        # Create input and output virtual streams params
        # Quantized argument signifies whether or not the incoming data is already quantized.
        # Data is quantized by HailoRT if and only if quantized == False .
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, quantized=False, format_type=FormatType.UINT8
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        # Define dataset params
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
        (
            self.image_height,
            self.image_width,
            self.channels,
        ) = self.input_vstream_info.shape

        self.postprocess = DetectionPostNMS()
        # create queues and processes
        self.input_queue = Queue(ctx=get_context())
        self.bypass_queue = Queue(ctx=get_context())
        self.output_queue = Queue(ctx=get_context())

        infer_process = Process(
            target=infer,
            args=(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
                self.network_group_params,
                self.postprocess.run,
                self.input_queue,
                self.output_queue,
                self.stop_event,
            ),
        )
        infer_process.start()

        # self.network_group.activate(self.network_group_params)

    def run(self, input_tensor):
        input_data = cv2.resize(input_tensor[0], (self.image_height, self.image_width))
        input_data = input_data[np.newaxis, :]
        self.input_queue.put(input_data)
        detections = self.output_queue.get()
        return detections
