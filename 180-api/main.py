from flask import Flask
from flask_cors import CORS

app = Flask(__name__)


app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return "Hello, World!"

if __name__ == '__main__':
    app.run()
