#!/usr/bin/env python
# -*- coding: utf-8 -*-
import zmq


def ServerSocket(port=5556):
    # opens server socket and returns it
    # use this socekt for send recv functions
    # see zmq doc for more options
    if not isinstance(port, int):
        raise TypeError("port should be integer")

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 10)  # limit Q size
    socket.bind("tcp://*:{}".format(port))
    return socket


def ClientSocket(port=5556, ip="127.0.0.1"):
    # opens client socket and returns it
    # use this socekt calling using send recv functions
    # only one server and one client is allowed
    # see zmq doc for more options
    if not isinstance(port, int):
        raise TypeError("port should be integer")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    # We must declare the socket as of type SUBSCRIBER, and pass a prefix filter.
    # Here, the filter is the empty string, which means we receive all messages.
    # We may subscribe to several filters, thus receiving from all.
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVHWM, 10)  # limit Q size
    socket.setsockopt(zmq.CONFLATE, 1)  # last msg only.
    socket.connect("tcp://{}:{}".format(ip, port))
    return socket


def SendData(data, socket):
    # recieves a python object and an open socket
    # To comunicate with none python process you should know what you are doing
    socket.send_pyobj(data)


def RecvData(socket):
    # recieves an open socket and returns a python object
    # To comunicate with none python process you should know what you are doing
    data = socket.recv_pyobj()
    return data


def SendJson(data, socket):
    # recieves a JSON string and an open socket
    socket.send_string(data)


def RecvJson(socket):
    # recieves an open socket and returns a JSON string
    data = socket.recv_string()
    return data


if __name__ == "__main__":
    import time

    socket = ClientSocket()
    print("Running ZMQ SUBSCRIBER")
    while 1:
        print(RecvData(socket))
        time.sleep(0.01)
