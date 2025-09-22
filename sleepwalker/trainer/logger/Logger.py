from abc import ABC, abstractmethod
from collections import defaultdict
from sleepwalker.utils import logger

import numpy as np

class Logger(ABC):

    def __init__(self, experiment_name, run_name):
        self.experiment_name = experiment_name
        self.run_name = run_name

    @abstractmethod
    def log(self, message):
        ...

    @abstractmethod
    def new_progress(self) -> Progress:
        ...

    @abstractmethod
    def log_metric(self, metric_name, metric_value):
        ...

    @abstractmethod
    def log_artifact(self, source_path, artifact_path):
        ...
