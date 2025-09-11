import mlflow

from sleepwalker.trainer.logger import Logger

class MLFlowLogger(Logger):

    def __init__(self, experiment_name, run_name, server_url):
        super().__init__(experiment_name, run_name)
        mlflow.set_tracking_uri(server_url)
        mlflow.set_experiment(experiment_name=experiment_name)
        mlflow.start_run(run_name=run_name)

    def log_metric(self, metric_name, metric_value, step=0):
        mlflow.log_metric(metric_name, metric_value, step=step)

    def close(self):
        mlflow.end_run()