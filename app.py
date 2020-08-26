import json
import time
import platform
import cv2
import imutils
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask import Response
from flask_assistant import Assistant, ask
from imutils.video import VideoStream
from pyngrok import ngrok
from pyngrok.conf import PyngrokConfig
from flask_caching import Cache

config = {
    #"DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "simple", # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}

coords_file = 'data.txt'
coords = None
try:
    with open('data.txt') as json_file:
        coords = json.load(json_file)
        print('co-ords loaded')
except FileNotFoundError:
    pass

WINDOWS = platform.system() == 'Windows'

cache = Cache(config={'CACHE_TYPE': 'simple'})
app = Flask(__name__)
app.config.from_mapping(config)
if WINDOWS:
    app.config["CACHE_TYPE"] = "null"

app.config['INTEGRATIONS'] = ['ACTIONS_ON_GOOGLE']
assist = Assistant(app, route='/', project_id='piparking-lauj')
cache.init_app(app)

port = 5000

if not WINDOWS:
    # Open a ngrok tunnel to the dev server
    print('setting up ngrok')
    pyngrok_config = PyngrokConfig(config_path='/home/pi/.ngrok2/ngrok.yml', region='eu')
    public_url = ngrok.connect(port, options={"subdomain": 'piparking'},
                               pyngrok_config=pyngrok_config)
    print(f" * ngrok tunnel {public_url} -> http://127.0.0.1:{port}")

    # Update any base URLs or webhooks to use the public ngrok URL
    app.config["BASE_URL"] = public_url

    app.config['INTEGRATIONS'] = ['ACTIONS_ON_GOOGLE']

    vs = VideoStream(src=0).start()
    time.sleep(1.0)

car_cascade = cv2.CascadeClassifier('cars.xml')


@app.route("/image")
@cache.cached(timeout=5)
def video_feed():
    frame = movie_frame()
    image = jpg(frame)
    return Response(image, mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/pc")
def pc():
    raw = cv2.imread('cars3.jpg')
    # raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    wide = imutils.auto_canny(raw)

    height, width = wide.shape

    marker_width = coords['width']
    for coord in coords['coords']:
        x = coord['x']
        y = coord['y']
        x_max = x + marker_width
        y_max = y + marker_width

        if x_max > width:
            x_max = width
        if y_max > height:
            y_max = height

        if y < 0:
            y = 0
        if x < 0:
            x = 0

        box = wide[y: y_max, x: x_max]

        n_white_pix = np.sum(box == 255)
        percent_white = n_white_pix / marker_width**2
        print(percent_white, n_white_pix)

    image = jpg(box)

    return Response(image, mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/car")
def car():
    frame = movie_frame()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    # To draw a rectangle in each cars
    for (x, y, w, h) in cars:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)

    image = jpg(gray)
    return Response(image, mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/calibrate", methods = ['GET', 'POST',])
def calibrate():
    if request.method == 'POST':
        global coords
        coords = request.get_json(silent=True)

        with open(coords_file, 'w') as outfile:
            json.dump(coords, outfile)

        print(coords) # Do your processing
        return jsonify({})
    return render_template("calibrate.html", title='Projects')

def movie_frame():
    if WINDOWS:
        return cv2.imread('cars3.jpg')
    frame = vs.read()
    return imutils.rotate(frame, 90)


def jpg(frame):
    (flag, encodedImage) = cv2.imencode(".jpg", frame)
    return b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'


@assist.action('Default Welcome Intent')
def parking():
    resp = ask("Here's a photo of the parking situation")
    resp.card(text='Parking situation',
              title='Cars',
              img_url='https://piparking.eu.ngrok.io/image',

              )

    return resp



if __name__ == '__main__':
    app.run()

