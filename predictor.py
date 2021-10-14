from prometheus_client import Gauge
from flask import Flask
from prometheus_flask_exporter import PrometheusMetrics
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import requests
import pandas as pd


def from_prometheus(query):
    """ query to prometheus using API """
    try:
        # XXX retry count를 지정해야 할지? > retry handler 사용하면 댐
        response = requests.get(url=os.environ['prom'], params={'query': query}, timeout=10)
    except Exception as e:
        # raise NewConnectionError if prometheus cannot be connectd
        # todo loggoer
        raise ConnectionError('failed to get from prometheus')
    metrics = response.json()['data']['result']
    for metric in metrics:
        print(metric)
        # todo smart 데이이터 어케 생겼는지 확인 필요
        # 지지고 볶고
    return None


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
    try:
        # get data from prometheus
        dataset = from_prometheus('collectd_memory')
        if dataset is None:
            raise ValueError('no data from prometheus')
        # get model      
        model = mlflow.sklearn.load_model('models:/smart-model/Production')
    except (ValueError, ConnectionError):
        app.logger.error('no recent smart data found')
        return 'no recent smart data found'
    except OSError:
        app.logger.error('no such file or directory')
        return 'no model found'
    # sklearn module required (not imported explicitly)
    result = model.predict(dataset)
    predict_result.set(result)
    return str(result)


def parse_env():
    # use get function instead of key access to avoid error
    host = os.environ.get('host')
    port = os.environ.get('port')
    repo = os.environ.get('repo')
    prom = os.environ.get('prom')

    # TODO regex validation
    if host is None:
        os.environ['host'] = '0.0.0.0'
    if port is None:
        os.environ['port'] = '9106'
    if repo is None:
        os.environ['repo'] = 'model-repository'
    if prom is None:
        os.environ['prom'] = "prometheus:9090"
    os.environ['prom'] = f"http://{os.environ['prom']}/api/v1/query"


if __name__ == '__main__':
    parse_env()
    mlflow.set_tracking_uri(f"http://{os.environ['repo']}:5000")
    app.run(host=os.environ.get('host'), port=os.environ.get('port'))