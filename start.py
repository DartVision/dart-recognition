from bottle import Bottle, run
from routes import *

app = Bottle()

run(app, host='0.0.0.0', port=80)
