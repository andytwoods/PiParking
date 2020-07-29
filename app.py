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


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route("/video_feed")
def video_feed():
    image = jpg()
    return Response(image, mimetype = "multipart/x-mixed-replace; boundary=frame")




def jpg():
    frame = vs.read()

    outputFrame = imutils.rotate(frame, 90)



    (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

    return b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'

jpg()

if __name__ == '__main__':
    app.run()
