import serial
import time
import fnmatch

CONNECT = "c"
DISCONNECT = "d"


class mqtt_arduino_client(object):
    def __init__(self) -> None:
        self.serial_port = None
        self.connected = self.connect()
        if not self.connected:
            print("ERROR: No Arduino found")
        self.detected = {
            "person": False,
            "banana": False,
            "cell phone": False,
        }

    def update(self, topic, payload):
        if topic in self.detected.keys():
            if payload != "0":  # payload is nyumber of detections
                self.detected[topic] = True
            else:
                self.detected[topic] = False
        if self.detected["banana"]:
            self.write("2")
        else:
            if self.detected["cell phone"]:
                self.write("3")
            else:
                if self.detected["person"]:
                    self.write("1")
                else:
                    self.write("0")
        print(f"status {self.detected}")

    def connect(self):
        self.available_ports = self.auto_detect_serial_unix()
        for port in self.available_ports:
            self.serial_port = serial.Serial(port, 115200, timeout=1)
            if self.check_connection():
                return True
            else:
                self.serial_port.close()
        return False

    def disconnect(self):
        self.write(DISCONNECT)
        self.serial_port.close()
        self.connected = False

    def auto_detect_serial_unix(self, preferred_list=["*"]):
        """try to auto-detect serial ports on win32"""
        import glob

        glist = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        ret = []

        # try preferred ones first
        for d in glist:
            for preferred in preferred_list:
                if fnmatch.fnmatch(d, preferred):
                    ret.append(d)
        if len(ret) > 0:
            return ret
        # now the rest
        for d in glist:
            ret.append(d)
        return ret

    def check_connection(self):
        self.write(CONNECT)
        time.sleep(0.05)
        data = self.read()
        if data == CONNECT:
            return True
        return False

    def write(self, data):  # data is a string
        if self.serial_port is None:
            return
        self.serial_port.write(bytes(data, "utf-8"))

    def read(self):
        if self.serial_port is None:
            return
        data = self.serial_port.read(self.serial_port.inWaiting())
        return data.decode("utf-8")


if __name__ == "__main__":
    arduino = mqtt_arduino_client()
    if not arduino.connected:
        exit()
    print("Starting script")
    while True:
        arduino.write("0")
        time.sleep(3)
        arduino.write("1")
        time.sleep(3)
        arduino.disconnect()
        time.sleep(5)
        arduino.connect()
