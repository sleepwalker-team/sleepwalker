"""Standalone entry point for the separately installed, compatible framework.

Run this file by absolute path, not with -m sleepwalker.server: module execution
would select the API checkout's older sleepwalker package.
"""

import json
from pathlib import Path
import signal
import sys
import traceback

import torch

from sleepwalker.deployment import load_packaged_model


def predict(request: dict, output: Path) -> None:
    if request["version"] != 1:
        raise ValueError("Unsupported worker protocol.")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    package_path = Path(request["package_path"])
    manifest = json.loads((package_path / "manifest.json").read_text(encoding="utf-8"))
    if request["sha256"] is not None and manifest["sha256"] != request["sha256"]:
        raise ValueError("Package changed since this job started; start the job again.")
    package = load_packaged_model(package_path, map_location=request["device"])
    # The package decides how it runs on one recording.
    frame = package.predict_patient(request["edf_path"], device=request["device"], batch_size=request["batch_size"], progress=False)
    if frame.empty:
        raise ValueError("No prediction windows were produced.")
    frame.to_csv(output / "predictions.csv", index=False)


def _exit_on_terminate(_signum, _frame) -> None:
    # Raising lets the framework's finally blocks remove its EDF cache in /dev/shm.
    raise SystemExit(1)


def main() -> None:
    signal.signal(signal.SIGTERM, _exit_on_terminate)
    request_path = Path(sys.argv[1])
    response = {"version": 1, "ok": False}
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        predict(request, request_path.parent)
        response["ok"] = True
    except (MemoryError, torch.OutOfMemoryError):
        traceback.print_exc()
        response["code"] = "RESOURCE_EXHAUSTED"
    except (OSError, KeyError, TypeError, ValueError):
        traceback.print_exc()
        response["code"] = "MODEL_INCOMPATIBLE"
    except Exception:
        traceback.print_exc()
        response["code"] = "INFERENCE_FAILED"
    (request_path.parent / "response.json").write_text(json.dumps(response), encoding="utf-8")
    if not response["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
