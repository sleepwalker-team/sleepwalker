#!/usr/bin/env python3
"""Download and convert pinned sleep foundation-model encoders."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleepwalker.datasets.Basedataset import ChannelConfig
from sleepwalker.datasets.UnlabelledDataset import UnlabelledDataset
from sleepwalker.deployment import save_packaged_model
from sleepwalker.models.PackagedEmbeddingModel import PackagedEmbeddingModel
from sleepwalker.models.preprocessors.FixedChannelStandardizer import FixedChannelStandardizer


SOURCES = {
    "sleepfm": {
        "repository": "https://github.com/zou-group/sleepfm-clinical.git",
        "revision": "2bcbae04c3592f61352addb7ac3d4193f0a3ca25",
        "checkpoint": "repository/sleepfm/checkpoints/model_base/best.pt",
        "checkpoint_sha256": "ffc9fc10233ebc4d0aae71abce87070db51b8bac6e4b16a5c1b4401a5f73f799",
    },
    "sleepgpt": {
        "repository": "https://github.com/LordXX505/SleepGPT.git",
        "revision": "fcbb2c7a6489b5742ab77d66483661e45a4aadf2",
        "checkpoint_url": "https://ndownloader.figshare.com/files/59576246",
        "checkpoint": "ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt",
        "checkpoint_md5": "c50548d45b6480701364dee0653b3eec",
    },
    "osf": {
        "repository": "https://github.com/yang-ai-lab/OSF-Open-Sleep-FM.git",
        "revision": "d7e4edbc77f2b72713402234036c72b98b9b83ca",
        "checkpoint_url": "https://huggingface.co/yang-ai-lab/OSF-Base/resolve/f2067f2d95b3bff43e4a690a471ba3d4457f12a7/osf_backbone.pth",
        "checkpoint": "osf_backbone.pth",
        "checkpoint_sha256": "c51190b1942556969af3c3d63c2e59430ddb1ea0377c50ea87df83712fc31857",
    },
}


class SleepFMClinicalEncoder(torch.nn.Module):
    """Expose the released modality-specific five-second embeddings as one vector."""

    def __init__(self, backbone, modality_channel_indices: list[list[int]]):
        super().__init__()
        if len(modality_channel_indices) != 4 or any(len(indices) == 0 for indices in modality_channel_indices):
            raise ValueError("SleepFM requires non-empty BAS, respiratory, ECG, and EMG channel groups.")
        self.backbone = backbone
        self.register_buffer("bas_indices", torch.tensor(modality_channel_indices[0], dtype=torch.long))
        self.register_buffer("resp_indices", torch.tensor(modality_channel_indices[1], dtype=torch.long))
        self.register_buffer("ecg_indices", torch.tensor(modality_channel_indices[2], dtype=torch.long))
        self.register_buffer("emg_indices", torch.tensor(modality_channel_indices[3], dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for indices in (self.bas_indices, self.resp_indices, self.ecg_indices, self.emg_indices):
            modality = x.index_select(1, indices)
            channel_mask = torch.zeros_like(modality[:, :, 0], dtype=torch.bool)
            _, contextual_embeddings = self.backbone(modality, channel_mask)
            embeddings.append(contextual_embeddings.flatten(start_dim=1))
        return torch.cat(embeddings, dim=1)


class OSFEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.model.to_tokens_2d(x)
        cls_token = self.model.cls_token.expand(tokens.size(0), -1, -1)
        features = torch.cat([cls_token, tokens], dim=1)
        features = features + self.model.pos_embedding[:, :features.size(1), :]
        features = self.model.dropout(features)
        for index in range(self.model.depth):
            features = getattr(self.model, f"block{index}")(features)
        features = self.model.norm(features)
        features = self.model.head(features)
        return features[:, 0]


class SleepGPTEncoder(torch.nn.Module):
    def __init__(self, model, input_channel_indices: list[int]):
        super().__init__()
        self.transformer = model.transformer
        self.token_type_embeddings = model.token_type_embeddings
        self.register_buffer("input_channel_indices", torch.tensor(input_channel_indices, dtype=torch.long))
        self.register_buffer("stft_window", torch.hann_window(self.transformer.patch_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_x = x.new_zeros((x.shape[0], self.transformer.max_channels, x.shape[2])).index_copy(1, self.input_channel_indices, x)
        frequency = self.get_fft(full_x)
        embedded = self.transformer.embed((full_x, frequency), attn_mask=None, mask=False)
        time_max_len = embedded["x_len"]
        features = embedded["x"]
        time_token_types = torch.zeros_like(features[:, :time_max_len, 0], dtype=torch.long)
        frequency_token_types = torch.ones_like(features[:, time_max_len:, 0], dtype=torch.long)
        time_features = features[:, :time_max_len] + self.token_type_embeddings(time_token_types)
        frequency_features = features[:, time_max_len:] + self.token_type_embeddings(frequency_token_types)
        features = torch.cat([time_features, frequency_features], dim=1)
        for block in self.transformer.blocks:
            features = block(features, mask=None, modality_type="tf", relative_position_bias=None)
        features = self.transformer.norm(features)
        time_features = features[:, 1:time_max_len].mean(dim=1)
        frequency_features = features[:, time_max_len + 1:].mean(dim=1)
        return torch.cat([time_features, frequency_features], dim=-1)

    def get_fft(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_samples = x.shape
        spectrum = torch.stft(x.reshape(batch_size * n_channels, n_samples), 256, self.transformer.hop_length, self.transformer.patch_size, self.stft_window, return_complex=True)
        magnitude = spectrum.abs()[:, :100, 1:]
        log_magnitude = (10 * torch.log10(magnitude + 1e-8)).transpose(-2, -1)
        mean = log_magnitude.mean(dim=-1, keepdim=True)
        standard_deviation = log_magnitude.std(dim=-1, keepdim=True)
        normalized = torch.where(standard_deviation != 0, (log_magnitude - mean) / standard_deviation, 0)
        return normalized.reshape(batch_size, n_channels, normalized.shape[-2], normalized.shape[-1])


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_digest(path: Path, algorithm: str, expected: str) -> None:
    actual = file_digest(path, algorithm)
    if actual != expected:
        raise ValueError(f"{path} {algorithm} mismatch: expected {expected}, got {actual}.")


def clone_repository(url: str, revision: str, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Repository destination already exists: {destination}")
    subprocess.run(["git", "clone", "--no-checkout", url, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)
    actual = subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if actual != revision:
        raise ValueError(f"Expected revision {revision}, got {actual}.")


def download_file(url: str, destination: Path, algorithm: str, expected: str) -> None:
    if destination.exists():
        verify_digest(destination, algorithm, expected)
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"Partial download already exists: {temporary}")
    urllib.request.urlretrieve(url, temporary)
    verify_digest(temporary, algorithm, expected)
    os.replace(temporary, destination)


def download(model_name: str, root: str | Path) -> Path:
    if model_name not in SOURCES:
        raise ValueError(f"Unknown foundation model {model_name!r}.")
    source = SOURCES[model_name]
    model_root = Path(root) / model_name
    if model_root.exists():
        raise FileExistsError(f"Model download directory already exists: {model_root}")
    model_root.mkdir(parents=True)
    clone_repository(source["repository"], source["revision"], model_root / "repository")
    if "checkpoint_url" in source:
        algorithm = "md5" if "checkpoint_md5" in source else "sha256"
        expected = source[f"checkpoint_{algorithm}"]
        download_file(source["checkpoint_url"], model_root / source["checkpoint"], algorithm, expected)
    else:
        verify_digest(model_root / source["checkpoint"], "sha256", source["checkpoint_sha256"])
    with (model_root / "source.json").open("w", encoding="utf-8") as handle:
        json.dump(source, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return model_root


def load_source(model_root: Path, model_name: str) -> dict:
    source_path = model_root / "source.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing download metadata: {source_path}")
    with source_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    if source != SOURCES[model_name]:
        raise ValueError(f"Downloaded source metadata for {model_name} does not match this converter.")
    revision = subprocess.run(["git", "-C", str(model_root / "repository"), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if revision != source["revision"]:
        raise ValueError(f"Expected source revision {source['revision']}, got {revision}.")
    checkpoint = model_root / source["checkpoint"]
    algorithm = "md5" if "checkpoint_md5" in source else "sha256"
    verify_digest(checkpoint, algorithm, source[f"checkpoint_{algorithm}"])
    return source


def trace_encoder(encoder: torch.nn.Module, example: torch.Tensor) -> tuple[torch.jit.ScriptModule, int]:
    encoder.eval()
    with torch.inference_mode():
        expected = encoder(example)
        if not isinstance(expected, torch.Tensor) or expected.ndim != 2:
            raise ValueError(f"Foundation encoder must return [B, E], got {type(expected).__name__} {getattr(expected, 'shape', None)}.")
        traced = torch.jit.trace(encoder, example, strict=False, check_trace=False)
        actual = traced(example)
    if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-4):
        raise ValueError("Traced encoder output does not match the upstream encoder.")
    return traced.eval(), int(expected.shape[1])


def load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def remove_data_parallel_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def convert_sleepfm(model_root: Path):
    upstream = load_file_module("sleepfm_clinical_conversion_models", model_root / "repository" / "sleepfm" / "models" / "models.py")
    with (model_root / "repository" / "sleepfm" / "checkpoints" / "model_base" / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    expected = {
        "model": "SetTransformer",
        "patch_size": 640,
        "embed_dim": 128,
        "sampling_duration": 5,
        "sampling_freq": 128,
        "modality_types": ["BAS", "RESP", "EKG", "EMG"],
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"Unexpected SleepFM base configuration: {mismatches}.")

    backbone = upstream.SetTransformer(
        config["in_channels"],
        config["patch_size"],
        config["embed_dim"],
        config["num_heads"],
        config["num_layers"],
        pooling_head=config["pooling_head"],
        dropout=0.0,
    )
    checkpoint = torch.load(model_root / SOURCES["sleepfm"]["checkpoint"], map_location="cpu", weights_only=False)
    if set(checkpoint) != {"state_dict"}:
        raise ValueError(f"Unexpected SleepFM checkpoint keys: {sorted(checkpoint)}.")
    backbone.load_state_dict(remove_data_parallel_prefix(checkpoint["state_dict"]), strict=True)

    channels = ["ECG", "EMG_Chin", "EMG_LLeg", "EMG_RLeg", "ABD", "THX", "NP", "SpO2", "SN", "EOG_E1_A2", "EOG_E2_A1", "EEG_C3_A2", "EEG_C4_A1"]
    modality_channel_indices = [list(range(9, 13)), list(range(4, 9)), [0], list(range(1, 4))]
    window_seconds = int(config["sampling_duration"] * 60)
    example = torch.zeros(1, len(channels), window_seconds * config["sampling_freq"])
    traced, embedding_dim = trace_encoder(SleepFMClinicalEncoder(backbone, modality_channel_indices), example)
    expected_embedding_dim = len(config["modality_types"]) * (window_seconds // 5) * config["embed_dim"]
    if embedding_dim != expected_embedding_dim:
        raise ValueError(f"SleepFM returned {embedding_dim} features, expected {expected_embedding_dim} contextual features.")
    return traced, embedding_dim, config["sampling_freq"], channels, None, window_seconds


def convert_osf(model_root: Path):
    sys.path.insert(0, str(model_root / "repository"))
    module = importlib.import_module("osf.backbone.vit1d_cls")
    payload = torch.load(model_root / SOURCES["osf"]["checkpoint"], map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    encoder = module.vit_base(num_leads=metadata["num_leads"], seq_len=metadata["seq_len"], patch_size=metadata["patch_size_time"], lead_wise=metadata["lead_wise"], patch_size_ch=metadata["patch_size_ch"])
    encoder.load_state_dict(payload["state_dict"])
    example = torch.zeros(1, int(metadata["num_leads"]), int(metadata["seq_len"]))
    traced, embedding_dim = trace_encoder(OSFEncoder(encoder), example)
    channels = ["ECG", "EMG_Chin", "EMG_LLeg", "EMG_RLeg", "ABD", "THX", "NP", "SN", "EOG_E1_A2", "EOG_E2_A1", "EEG_C3_A2", "EEG_C4_A1"]
    if len(channels) != int(metadata["num_leads"]):
        raise ValueError("OSF checkpoint channel count does not match the published channel order.")
    return traced, embedding_dim, int(round(int(metadata["seq_len"]) / 30)), channels, torch.nn.Hardtanh(-6.0, 6.0), 30


def convert_sleepgpt(model_root: Path):
    sys.path.insert(0, str(model_root / "repository"))
    module = importlib.import_module("main.modules.backbone_pretrain")
    checkpoint = model_root / SOURCES["sleepgpt"]["checkpoint"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = dict(payload["hyper_parameters"]["config"])
    if config["mode"] != "pretrain":
        raise ValueError(f"Expected a SleepGPT pretraining checkpoint, got mode {config['mode']!r}.")
    # The published checkpoint predates these upstream config fields and contains a machine-local load_path.
    config["visual"] = False
    config["stage"] = False
    config["load_path"] = ""
    config["actual_channels"] = "physio"
    config["loss_names"] = dict(config["loss_names"])
    config["loss_names"]["Spindle"] = 0
    model = module.Model_Pre(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    # The upstream in-place assignment otherwise bakes the example batch size into the traced graph.
    model.transformer.cls_token_pos_embed = torch.nn.Parameter(model.transformer.cls_token_pos_embed.flatten())
    pretrained_channels = ["C3", "C4", "EMG", "EOG", "F3", "Fpz", "O1", "Pz"]
    if int(config["random_choose_channels"]) != len(pretrained_channels):
        raise ValueError("SleepGPT checkpoint does not use the published eight-channel pretraining layout.")
    input_channel_indices = [0, 1, 2, 3, 4, 6]
    channels = [pretrained_channels[index] for index in input_channel_indices]
    example = torch.zeros(1, len(channels), 30 * 100)
    traced, embedding_dim = trace_encoder(SleepGPTEncoder(model, input_channel_indices), example)
    pretrained_means = [-0.068460, 0.191040, 0.389370, -2.093800, 0.0016496, -0.000048439, 0.00081125, 0.000071748]
    pretrained_standard_deviations = [34.6887, 34.9556, 23.2826, 35.4035, 26.8738, 4.9272, 25.1366, 3.6142]
    means = [pretrained_means[index] for index in input_channel_indices]
    standard_deviations = [pretrained_standard_deviations[index] for index in input_channel_indices]
    return traced, embedding_dim, 100, channels, FixedChannelStandardizer(means, standard_deviations), 30


def convert(model_name: str, model_root: str | Path, destination: str | Path):
    model_root = Path(model_root)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Package destination already exists: {destination}")
    source = load_source(model_root, model_name)
    converters = {"sleepfm": convert_sleepfm, "sleepgpt": convert_sleepgpt, "osf": convert_osf}
    encoder, embedding_dim, sample_frequency, channel_names, preprocessor, window_seconds = converters[model_name](model_root)
    ts_len = window_seconds * sample_frequency
    preprocessors = [] if preprocessor is None else [preprocessor]
    model = PackagedEmbeddingModel(encoder=encoder, embedding_dim=embedding_dim, ts_len=ts_len, n_channels=len(channel_names), channel_first=True, preprocessors=preprocessors)
    input_unit = "uV" if model_name == "sleepgpt" else None
    dataset = UnlabelledDataset(
        channels=[ChannelConfig(logical_name=name, physical_names=[name], unit=input_unit) for name in channel_names],
        sample_frequency=sample_frequency,
        resample_type="polyphase" if model_name == "sleepfm" else "nearest",
        total_input=f"{window_seconds}s",
        stride=f"{window_seconds}s",
        z_normalize=model_name in {"osf", "sleepfm"},
        assume_units_if_missing=True,
    )
    return save_packaged_model(destination, name=model_name, model=model, dataset=dataset, classification_contract=None, task=None, config={"foundation_model": model_name, "source": source})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("model", choices=sorted(SOURCES))
    download_parser.add_argument("root")
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("model", choices=sorted(SOURCES))
    convert_parser.add_argument("model_root", help="The model-specific directory created by download.")
    convert_parser.add_argument("destination", help="New PackagedModel directory.")
    args = parser.parse_args()
    if args.command == "download":
        download(args.model, args.root)
    else:
        convert(args.model, args.model_root, args.destination)


if __name__ == "__main__":
    main()
