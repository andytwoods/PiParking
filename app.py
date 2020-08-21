import time

import cv2
import imutils
from flask import Flask
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

cache = Cache(config={'CACHE_TYPE': 'simple'})
app = Flask(__name__)
app.config.from_mapping(config)
app.config['INTEGRATIONS'] = ['ACTIONS_ON_GOOGLE']
assist = Assistant(app, route='/', project_id='piparking-lauj')
cache.init_app(app)

port = 5000

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


@app.route("/image")
@cache.cached(timeout=5)
def video_feed():
    frame = movie_frame()
    image = jpg(frame)
    return Response(image, mimetype="multipart/x-mixed-replace; boundary=frame")


def movie_frame():
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

