from flask import Flask
from pyngrok import ngrok
from pyngrok.conf import PyngrokConfig
app = Flask(__name__)

port = 5000

# Open a ngrok tunnel to the dev server
pyngrok_config = PyngrokConfig(config_path='/home/pi/.ngrok2/ngrok.yml', region='eu')
public_url = ngrok.connect(port, options={"subdomain": 'piparking'},
                           pyngrok_config=pyngrok_config)
print(f" * ngrok tunnel {public_url} -> http://127.0.0.1:{port}" )

# Update any base URLs or webhooks to use the public ngrok URL
app.config["BASE_URL"] = public_url


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
