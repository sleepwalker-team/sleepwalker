class Logger:

    def __init__(self, experiment_name, run_name):
        self.experiment_name = experiment_name
        self.run_name = run_name

    def log_metric(self, metric_name, metric_value, step=0):
        print(f'{step} - {metric_name} - {metric_value:.3f}')

    def close(self):
        return