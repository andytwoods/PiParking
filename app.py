import os

import imutils
from flask import Flask
from pyngrok import ngrok
from pyngrok.conf import PyngrokConfig
from flask import Response
from flask import Flask
from flask import render_template
import threading
from imutils.video import VideoStream
import time
import cv2
import numpy as np


app = Flask(__name__)

port = 5000

# Open a ngrok tunnel to the dev server
pyngrok_config = PyngrokConfig(config_path='/home/pi/.ngrok2/ngrok.yml', region='eu')
public_url = ngrok.connect(port, options={"subdomain": 'piparking'},
                           pyngrok_config=pyngrok_config)
print(f" * ngrok tunnel {public_url} -> http://127.0.0.1:{port}")

# Update any base URLs or webhooks to use the public ngrok URL
app.config["BASE_URL"] = public_url
vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
     "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
     "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print("[INFO] loading model…")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route("/video_feed")
def video_feed():
    frame = movie_frame()
    image = jpg(frame)
    return Response(image, mimetype="multipart/x-mixed-replace; boundary=frame")

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None

    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image

    resized = cv2.resize(image, dim)

    # return the resized image
    return resized


@app.route("/car")
def car():
    frame = movie_frame()
    frame = resize(frame, width=300)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
    net.setInput(blob, scalefactor=1.0 / 127.5, mean=[127.5,127.5, 127.5])

    detections = net.forward()

    image = jpg(frame)
    return Response(image, mimetype="multipart/x-mixed-replace; boundary=frame")


def movie_frame():
    frame = vs.read()
    return imutils.rotate(frame, 90)


def jpg(frame):
    (flag, encodedImage) = cv2.imencode(".jpg", frame)
    return b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'


if __name__ == '__main__':
    app.run()
