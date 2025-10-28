#!/bin/bash

TRACKING_URI=`cat ~/mlflow/auth_config.ini`
ARTIFACT_URI="/raid/mlruns"
PORT=5001

mlflow ui --backend-store-uri $TRACKING_URI --default-artifact-root $ARTIFACT_URI --port $PORT