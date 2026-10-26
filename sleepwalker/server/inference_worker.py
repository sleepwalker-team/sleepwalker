"""Standalone entry point for the separately installed, compatible framework.

Run this file by absolute path, not with -m sleepwalker.server: module execution
would select the API checkout's older sleepwalker package.
"""

import json
import math
from pathlib import Path
import sys
import traceback

import pandas as pd
import torch

from sleepwalker.deployment import load_packaged_model
from sleepwalker.datasets.EDFCache import EDFCache


def predict(request: dict, output: Path) -> None:
    if request["version"] != 1:
        raise ValueError("Unsupported worker protocol.")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    package_path = Path(request["package_path"])
    manifest = json.loads((package_path / "manifest.json").read_text(encoding="utf-8"))
    if request["sha256"] is not None and manifest["sha256"] != request["sha256"]:
        raise ValueError("Package changed since API registry initialization; restart the API.")
    package = load_packaged_model(package_path, map_location=request["device"])
    contract = package.classification_contract
    if contract is None or contract.get("type") != "single-head-multiclass":
        raise ValueError("Only single-head classifiers are enabled.")
    resolution = pd.to_timedelta(contract["target_resolution"]).total_seconds() / contract["sequence_len"]
    if package.task != request["task"] or list(contract["classes"]) != request["classes"]:
        raise ValueError("Executable output contract differs from API metadata.")
    if not math.isclose(resolution, request["resolution"], rel_tol=1e-9):
        raise ValueError("Executable output resolution differs from API metadata.")
    if list(package.dataset.get_input_channels()) != request["input_channels"]:
        raise ValueError("Executable input contract differs from API metadata.")
    # Same public preparation path as predict_patient(), with an explicitly
    # owned cache directory. The API parent removes it even after SIGKILL.
    dataset = package.dataset.clone()
    package.assert_compatible(dataset)
    dataset.set_rejection_strategy("none")
    dataset.initialize([request["edf_path"]], num_workers=0, strict=True)
    if dataset.get_n_patients() != 1:
        raise ValueError("Could not prepare exactly one EDF recording.")
    with EDFCache(1, directory=output) as cache:
        dataset.set_edf_cache(cache)
        try:
            frame = package.predict_dataset(
                dataset, device=request["device"], batch_size=request["batch_size"],
                num_workers=0, n_repeat=1, rejection_strategy="none", progress=False, edf_cache_patients=0,
            )
        finally:
            dataset.set_edf_cache(None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("No prediction windows were produced.")
    frame.to_csv(output / "predictions.csv", index=False)


def main() -> None:
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
