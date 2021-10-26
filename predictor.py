from flask.scaffold import F
from prometheus_client import Gauge
from flask import Flask, request
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client.core import GaugeMetricFamily
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pyfunc
import os
import requests
import pandas as pd
from xgboost import DMatrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time


RF = 'RandomForestClassifier'
XGB = 'XGBClassifier'
OCSVM = 'OneClassSVM'
MODELS = {
    RF: None,
    XGB: None,
    OCSVM: None   
}
FEATURES = ['read-error-rate-normal',
            'read-error-rate-raw',
            'throughput-performance-normal',
            'throughput-performance-raw',
            'spin-up-time-normal',
            'spin-up-time-raw',
            'start/stop-count-normal',
            'start/stop-count-raw',
            'reallocated-sectors-count-normal',
            'reallocated-sectors-count-raw',
            'seek-error-rate-normal',
            'seek-error-rate-raw',
            'seek-time-performance-normal',
            'seek-time-performance-raw',
            'power-on-hours-normal',
            'power-on-hours-raw',
            'spin-retry-count-normal',
            'spin-retry-count-raw',
            'recalibration-retries-normal',
            'recalibration-retries-raw',
            'power-cycle-count-normal',
            'power-cycle-count-raw',
            'soft-read-error-rate-normal',
            'soft-read-error-rate-raw',
            'current-helium-level-normal',
            'current-helium-level-raw',
            'available-reserved-space-normal',
            'available-reserved-space-raw',
            'ssd-wear-leveling-count-normal',
            'ssd-wear-leveling-count-raw',
            'unexpected-power-loss-count-normal',
            'unexpected-power-loss-count-raw',
            'power-loss-protection-failure-normal',
            'power-loss-protection-failure-raw',
            'wear-range-delta-normal',
            'wear-range-delta-raw',
            'used-reserved-block-count-total-normal',
            'used-reserved-block-count-total-raw',
            'unused-reserved-block-count-total-normal',
            'unused-reserved-block-count-total-raw',
            'program-fail-count-total-normal',
            'program-fail-count-total-raw',
            'erase-fail-count-normal',
            'erase-fail-count-raw',
            'sata-downshift-error-count-normal',
            'sata-downshift-error-count-raw',
            'end-to-end-error-normal',
            'end-to-end-error-raw',
            'reported-uncorrectable-errors-normal',
            'reported-uncorrectable-errors-raw',
            'command-timeout-normal',
            'command-timeout-raw',
            'high-fly-writes-normal',
            'high-fly-writes-raw',
            'temperature-difference-normal',
            'temperature-difference-raw',
            'g-sense-error-rate-normal',
            'g-sense-error-rate-raw',
            'power-off-retract-count-normal',
            'power-off-retract-count-raw',
            'load-cycle-count-normal',
            'load-cycle-count-raw',
            'temperature-normal',
            'temperature-raw',
            'hardware-ecc-recovered-normal',
            'hardware-ecc-recovered-raw',
            'reallocation-event-count-normal',
            'reallocation-event-count-raw',
            'current-pending-sector-count-normal',
            'current-pending-sector-count-raw',
            '(offline)-uncorrectable-sector-count-normal',
            '(offline)-uncorrectable-sector-count-raw',
            'ultradma-crc-error-count-normal',
            'ultradma-crc-error-count-raw',
            'multi-zone-error-rate-normal',
            'multi-zone-error-rate-raw',
            'data-address-mark-errors-normal',
            'data-address-mark-errors-raw',
            'flying-height-normal',
            'flying-height-raw',
            'vibration-during-write-normal',
            'vibration-during-write-raw',
            'disk-shift-normal',
            'disk-shift-raw',
            'loaded-hours-normal',
            'loaded-hours-raw',
            'load/unload-retry-count-normal',
            'load/unload-retry-count-raw',
            'load-friction-normal',
            'load-friction-raw',
            'load/unload-cycle-count-normal',
            'load/unload-cycle-count-raw',
            "load-'in'-time-normal",
            "load-'in'-time-raw",
            'life-left-(ssds)-normal',
            'life-left-(ssds)-raw',
            'endurance-remaining-normal',
            'endurance-remaining-raw',
            'media-wearout-indicator-(ssds)-normal',
            'media-wearout-indicator-(ssds)-raw',
            'average-erase-count-and-maximum-erase-count-normal',
            'average-erase-count-and-maximum-erase-count-raw',
            'good-block-count-and-system(free)-block-count-normal',
            'good-block-count-and-system(free)-block-count-raw',
            'head-flying-hours-normal',
            'head-flying-hours-raw',
            'total-lbas-written-normal',
            'total-lbas-written-raw',
            'total-lbas-read-normal',
            'total-lbas-read-raw',
            'read-error-retry-rate-normal',
            'read-error-retry-rate-raw',
            'minimum-spares-remaining-normal',
            'minimum-spares-remaining-raw',
            'newly-added-bad-flash-block-normal',
            'newly-added-bad-flash-block-raw',
            'free-fall-protection-normal',
            'free-fall-protection-raw']


class SmartPredictorExporter(object):
    
    def __init__(self, prometheus_url, logger):
        self.url = prometheus_url
        self.logger = logger
        self.name_predict = 'ampm_smart_failure'
        self.name_predict_proba = 'ampm_smart_failure_proba'
        self.desc_predict = "Predictor_exporter: 'predict'  Type: 'smart_attribute' Dstype: 'api.Gauge'"
        self.desc_predict_proba = "Predictor_exporter: 'predict_proba'  Type: 'smart_attribute' Dstype: 'api.Gauge'"
        self.labels = ['exported_instance', 'model', 'drive']
        self.queries = ['collectd_smart_smart_attribute_current', 'collectd_smart_smart_attribute_pretty']

    def collect(self):
        predict_metric = GaugeMetricFamily(name=self.name_predict, documentation=self.desc_predict, labels=self.labels)
        predict_proba_metric = GaugeMetricFamily(name=self.name_predict_proba, documentation=self.desc_predict_proba, labels=self.labels)

        datasets = self.from_prometheus()
        if datasets is None:
            datasets = dict()

        # by exported_instance
        for instance_name in datasets:
            drives = datasets[instance_name]
            # by drive
            for drive_name in drives:
                smart_data = pd.DataFrame([self.add_features(drives[drive_name])])

                if smart_data.empty:
                    self.logger.error(f"failed to convert data from '{instance_name}/{drive_name}' to DataFrame")
                    continue
                
                self.add_metrics(predict_metric, predict_proba_metric, instance_name, drive_name, smart_data)

        yield predict_metric
        yield predict_proba_metric

    
    def add_metrics(self, predict_metric, predict_proba_metric, instance_name, drive_name, smart_data):
        for model_name in MODELS:
            model = MODELS[model_name]
            try:
                if model is None:
                    raise AttributeError('Model Not Loaded')
                mod_data = self.cast_data(model_name, smart_data)
                result = model.predict(mod_data)[0]
                predict = result if model == RF else (1 if result > 0.5 else 0)
                predict_proba = model.predict_proba(mod_data)[0][1] if model == RF else result
                predict_metric.add_metric([instance_name, model_name, drive_name], predict)
                predict_proba_metric.add_metric([instance_name, model_name, drive_name], predict_proba)
            except (ValueError, AttributeError, TypeError, mlflow.exceptions.MlflowException) as e:
                self.logger.error(f"'{model_name}' : {e}")

    def add_features(self, metrics):
        for feature in FEATURES:
            # attribute_name = feature[:(len(feature) - feature[::-1].index('-') - 1)]
            from_metrics = metrics.get(feature)
            if from_metrics is None:
                metrics[feature] = 0
        return metrics

    def cast_data(self, model_name, smart_data):
        if model_name == XGB:
            return smart_data
            # return DMatrix(smart_data)
        elif model_name == OCSVM:
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
        for metric in metrics:
            server_name = metric['metric']['instance']
            smart = instances_smart.get(server_name)
            if smart is None:
                instances_smart[server_name] = dict()
                smart = instances_smart.get(server_name)
            
            drive_name = metric['metric']['smart']
            drive = smart.get(drive_name)
            if drive is None:
                smart[drive_name] = dict()
                drive = smart.get(drive_name)

            attribute_name = metric['metric']['type']
            if not attribute_name.endswith('raw') and not attribute_name.endswith('normal'):
                attribute_name += '-raw' if metric['metric']['__name__'].endswith('pretty') else '-normal'
            # value = [milliseconds, value]
            drive[attribute_name] = float(metric['value'][1])    
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


def load_model(max_loop, timesleep):
    mlflow.set_tracking_uri(os.environ['repo'])
    cnt = 0
    while cnt < max_loop:
        try:
            for model_name in MODELS:
                ref = f'models:/{model_name}/Production'
                MODELS[model_name] = mlflow.pyfunc.load_model(ref)
            break
        except Exception as e:
            print(f"'{model_name}' ({cnt+1} tried): {e}")
            time.sleep(timesleep)
            cnt += 1


if __name__ == '__main__':
    parse_env()
    load_model(3, 5)
    # mlflow.set_tracking_uri(os.environ['repo'])
    exporter.registry.register(SmartPredictorExporter(os.environ['prom'], app.logger))
    app.run(host=os.environ.get('host'), port=os.environ.get('port'))