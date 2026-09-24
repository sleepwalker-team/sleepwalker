# Download Sleep-EDFx

[Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) contains 197 whole-night polysomnographic recordings with technician-scored hypnograms. The recordings include EEG, EOG, chin EMG, and event markers. Some recordings also include respiration and body temperature.

Create the destination directory and run the downloader included with Sleep-EDFx:

!!! note
    If you have Sleep-EDFx already downloaded, then you can skip this part. Just make sure that you did not change the filenames of individual recordings. Otherwise we cannot match the corresponding Hypnogram files to the patient recordings

```bash
mkdir -p data/sleep-edfx
python -m sleepwalker.datasets.SleepEDFx --out data/sleep-edfx
```

The command downloads both Sleep Cassette (`SC`) and Sleep Telemetry (`ST`) and checks every downloaded file against the SHA256 digest stored in Sleepwalker:

```text
data/sleep-edfx/
├── SC/
│   ├── SC4001E0-PSG.edf
│   ├── SC4001EC-Hypnogram.edf
│   └── ...
└── ST/
    ├── ST7011J0-PSG.edf
    ├── ST7011JP-Hypnogram.edf
    └── ...
```

!!! warning "No Python download API yet"
    The downloader is currently a module command, not a supported importable function. A public function with tested subset selection, return values, resuming, and failure behavior is tracked on the [roadmap](../roadmap.md#add-a-public-sleep-edfx-download-api).

## Find the recordings

Use [`get_edf_files_in_repo()`](../reference/api.md#get-edf-files-in-repo) to list edf files inside a folder (sometimes referred to as "repository"):

```python
from sleepwalker.datasets.utils import get_edf_files_in_repo

files = get_edf_files_in_repo("data/sleep-edfx/SC")
recordings = sorted(path for path in files if path.endswith("-PSG.edf"))

if not recordings:
    raise FileNotFoundError("No Sleep-EDFx PSG recordings found")

print(f"Found {len(recordings)} recordings")
print(recordings[0])
```

Continue with [Train and export a sleep-staging model](train-sleep-staging.md), or read [Load and prepare data](data.md) to understand how windows, labels, and normalization work first.
