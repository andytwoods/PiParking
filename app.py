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

app = Flask(__name__)

port = 5000

# Open a ngrok tunnel to the dev server
pyngrok_config = PyngrokConfig(config_path='/home/pi/.ngrok2/ngrok.yml', region='eu')
public_url = ngrok.connect(port, options={"subdomain": 'piparking'},
                           pyngrok_config=pyngrok_config)
print(f" * ngrok tunnel {public_url} -> http://127.0.0.1:{port}")

# Update any base URLs or webhooks to use the public ngrok URL
app.config["BASE_URL"] = public_url
vs = VideoStream(src=0).start()
time.sleep(1.0)

car_cascade = cv2.CascadeClassifier('cars.xml')


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route("/video_feed")
def video_feed():
    frame = movie_frame()
    image = jpg(frame)
    return Response(image, mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/car")
def car():
    frame = movie_frame()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    # To draw a rectangle in each cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

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
