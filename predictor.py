from prometheus_client import Gauge
from flask import Flask, request
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client.core import GaugeMetricFamily
import mlflow.sklearn
import mlflow.xgboost
import os
import requests
import pandas as pd
from xgboost import DMatrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class SmartPredictorExporter(object):
    
    def __init__(self, prometheus_url, logger):
        self.url = prometheus_url
        self.logger = logger
        self.name_predict = 'ampm_smart_failure'
        self.name_predict_proba = 'ampm_smart_failure_proba'
        self.desc_predict = "Predictor_exporter: 'predict'  Type: 'smart_attribute' Dstype: 'api.Gauge'"
        self.desc_predict_proba = "Predictor_exporter: 'predict_proba'  Type: 'smart_attribute' Dstype: 'api.Gauge'"
        self.labels = ['pc-name', 'model']
        self.queries = ['collectd_smart_smart_attribute_current', 'collectd_smart_smart_attribute_pretty']
        self.models = ['RandomForestClassifier', 'XGBClassifier', 'OneClassSVM']

    def collect(self):
        result = GaugeMetricFamily(name=self.name_predict, documentation=self.desc_predict, labels=self.labels)
        result_proba = GaugeMetricFamily(name=self.name_predict_proba, documentation=self.desc_predict_proba, labels=self.labels)
        datasets = self.from_prometheus()

        if datasets is None:
            datasets = dict()

        # get smart data by instance
        for instance in datasets:
            metrics = datasets[instance]
            smart_data = pd.DataFrame([metrics])

            if smart_data.empty:
                self.logger.error(f"failed to convert data from '{instance}' to DataFrame")
                continue
            # predict failure 
            for model_name in self.models:
                try:
                    ref = f'models:/{model_name}/Production'
                    model = mlflow.xgboost.load_model(ref) if model_name == 'XGBClassifier' else mlflow.sklearn.load_model(ref)
                    mode_data = self.modify_data(model_name, smart_data)
                    predict = model.predict(mode_data)
                    predict_proba = -1
                    if model_name == 'XGBClassifier':
                        predict_proba = predict[0]
                        predict = 1 if predict_proba > 0.5 else 0
                    else:
                        predict = predict[0]
                        predict_proba = model.predict_proba(mode_data)[0][1]
                    result.add_metric([instance, model_name], predict)
                    if predict_proba != -1:
                        result_proba.add_metric([instance, model_name], predict_proba)
                except OSError:
                    self.logger.error(f"No '{model_name}' model found")
                except (ValueError, AttributeError, TypeError) as e:
                    self.logger.error(f"'{model_name}' : {e}")
        yield result
        yield result_proba

    def modify_data(self, model_name, smart_data):
        if model_name == 'XGBClassifier':
            return DMatrix(smart_data)
        elif model_name == 'OneClassSVM':
            return PCA(n_components=1).fit_transform(StandardScaler().fit_transform(smart_data))
        else:
            return smart_data

    def from_prometheus(self):
        metrics = list()
        try:
            for query in self.queries:
                response = requests.get(url=self.url, params={'query': query}, timeout=10)
                metrics.extend(response.json()['data']['result'])
            if not metrics:
                raise ValueError('No result from prometheus')
        except Exception as e:
            self.logger.error(e)
            return None
        
        instances_smart = dict()
        # instance 구분
        for metric in metrics:
            instance = metric['metric']['instance']
            smart = instances_smart.get(instance)
            if smart is None:
                instances_smart[instance] = dict()
                smart = instances_smart.get(instance)
            attribute_name = metric['metric']['type']
            attribute_name += '-raw' if metric['metric']['__name__'].endswith('pretty') else '-normal'
            # value = [milliseconds, value]
            self.logger.error(type(metric['value'][1]))
            smart[attribute_name] = float(metric['value'][1])
        
        return instances_smart


app = Flask(__name__)
exporter = PrometheusMetrics(app)


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
        os.environ['repo'] = 'http://model-repository:5000'
    if prom is None:
        os.environ['prom'] = "prometheus:9090"
    os.environ['prom'] = f"http://{os.environ['prom']}/api/v1/query"


if __name__ == '__main__':
    parse_env()
    mlflow.set_tracking_uri(os.environ['repo'])
    exporter.registry.register(SmartPredictorExporter(os.environ['prom'], app.logger))
    app.run(host=os.environ.get('host'), port=os.environ.get('port'))