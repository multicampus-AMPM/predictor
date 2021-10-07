#!/usr/bin/env python3

from prometheus_client import Gauge
from flask import Flask
from prometheus_flask_exporter import PrometheusMetrics
import mlflow.sklearn
import pandas as pd
import os


app = Flask(__name__)
exporter = PrometheusMetrics(app)
predict_result = Gauge('ampm_predict_result', 'predict result on the reqeust smart data')


@app.route('/favicon.ico')
@exporter.do_not_track()
def favicon():
    return 'ok'


@app.route('/')
@exporter.do_not_track()
def main():
    """ context root """
    return """
        <html>
            <head><title>Predictor Exporter</title></head>
            <body>
                <h1>Predictor Exporter</h1>
                <p><a href='/metrics'>Metrics</a></p>
            </body>
        </html>
    """


@app.route('/predict/smart')
@exporter.do_not_track()
def predict():
    # get data from prometheus
    X = pd.read_json('{"Raw Read Error Rate":{"560631":0.451613},"SpinUpTime":{"560631":1.0},"Reallocated Sector Count":{"560631":1.0},"Seek Error Rate":{"560631":0.536585},"Power on Hours":{"560631":0.052632},"Reported Uncorrectable Error":{"560631":1.0},"High Fly Writes":{"560631":1.0},"Temperature Celsius":{"560631":-0.6},"Hardware ECC Recovered":{"560631":-0.225806},"Current Pending Sector":{"560631":1.0},"Reallocated Sectors Count":{"560631":-1.0},"Current Pending Sectors counts":{"560631":-1.0}}')
    # get model
    try:
        model = mlflow.sklearn.load_model(os.path.join(os.environ['repo'], 'smart-model'))
    except OSError:
        # no such file or directory
        # TODO logging
        return 'no model found'
    # sklearn module required (not imported explicitly)
    predict_result.set(model.predict(X))
    return 'ok'


def parse_env():
    # use get function instead of key reference to avoid error
    host = os.environ.get('host')
    port = os.environ.get('port')
    repo = os.environ.get('repo')

    # TODO regex validation
    if host is None:
        os.environ['host'] = '0.0.0.0'
    if port is None:
        os.environ['port'] = '9106'
    if repo is None:
        os.environ['repo'] = '/ampm/models/'


if __name__ == '__main__':
    parse_env()
    app.run(host=os.environ.get('host'), port=os.environ.get('port'))