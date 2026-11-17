# Performance

Start with random window loading and measure the complete training or prediction
path. A slow isolated loader does not necessarily slow the model: workers can
prepare the next batch while the current batch executes. Conversely, fast model
execution can expose input preparation. Disk I/O, CPU preparation, startup and
model computation can each dominate; none is the universal bottleneck.

## What loading actually does

Initialization creates recording descriptors, event indices, candidate offsets
and per-channel processor state. Processors with a recording-level `fit` also
read signals during initialization. That work can be substantial.

For an accepted window, the ordinary path reads the requested native channel
intervals, aligns/resamples them, applies window-level preprocessing, and builds
a tensor sample. Targets are prepared before signal reads so target rejection
can avoid signal work. Signal-quality rejection can occur later. See
[loading data](data.md) for the processing contract.

The direct reader opens and closes a pyedflib reader for each native request.
Multiple read modes or fallback aliases can require additional requests.
An open edflib handle retains parsed header/calibration state; closing it discards
that state. File reads can still hit C stdio buffers or the OS page cache, so
reopening a recording does **not** imply a physical disk read for every window.
EDF sample payload bytes are not a measurement of storage traffic.

Whole-recording filtering is not interchangeable with window filtering: boundary
conditions and window statistics can differ. Keep preparation semantics fixed
when comparing execution choices.

## Measure exposed waiting, not just loader speed

Record initialization and model warm-up separately from epochs. Within an epoch,
measure wall time, accepted windows/optimizer updates, first-batch waiting and
waiting for later batches. Compare complete prediction calls including worker
startup and result formatting. GPU activity alone does not identify a bottleneck.

In a short A30 investigation on 2026-10-25, native training used eight loader
workers, batch 128, 16 recordings, and two epochs of 4,096 candidate windows:

| Second epoch | Total time | First-batch wait | Wait for remaining 31 batches |
| --- | ---: | ---: | ---: |
| Sleep workload | 14.58 s | 1.53 s | 8.2 ms total |
| Arousal workload | 5.23 s | 1.05 s | 4.7 ms total |

These are diagnostic timings, not universal performance figures. The cohort was
small, OS caches were not flushed, and initialization/model warm-up are excluded.
They show that most input work was overlapped after the first batch in those
runs. Larger cohorts, fewer workers, different storage or a faster model may
expose preparation costs.

Separate warm-file CPU profiles found the following costs per accepted sequential
window. These retained the existing models' input preparation:

| Operation | Sleep | Arousal |
| --- | ---: | ---: |
| Complete window preparation | 10.95 ms | 7.69 ms |
| Resampling and selection | 5.00 ms | 2.18 ms |
| Channel preprocessing | 1.25 ms | 1.99 ms |
| Explicit timestamp-index construction | 0.25 ms | 0.15 ms |
| Final sample/tensor callback | 0.13 ms | 0.04 ms |
| EDF reader construction | 0.056 ms | 0.040 ms |

The stage timers came from a separate instrumented pass; these rows are not a
complete additive breakdown. Resampling includes necessary computation and
internal index handling, not only removable overhead. DataFrame `to_numpy()` and
`torch.from_numpy()` shared memory in the measured cases; particular selections,
explicit copies and dtype changes allocated. Replacing DataFrames wholesale has
no demonstrated end-to-end benefit. Profile concrete operations before refactoring.

## Workers and threads

Initialization workers and loader workers do different jobs. The former prepare
recordings; the latter materialize samples during execution. Tune them separately.
`build_loader` defaults to persistent training workers and prefetch factor two
when workers are enabled. Pinning and persistence are explicit options.

```python
from sleepwalker.datasets.Basedataset import batch_collate
from sleepwalker.training.loader import build_loader

loader = build_loader(
    dataset,
    batch_size=128,
    num_workers=4,  # starting point to measure, not an optimum
    n_samples=50_000,
    collate_fn=batch_collate,
    sampling="random",
    seed=17,
)
```

Try a small range of worker counts and inspect throughput and memory. Increasing
workers can improve overlap or create CPU/memory contention. If a window costs
`t_window` and a batch has `B` windows, the idealized supply time is approximately
`B * t_window / W` for `W` workers. Compare that with the consumer's batch time;
this ignores coordination, variable sample costs and shared-resource contention.

Numeric libraries may use additional threads. Where oversubscription occurs,
set `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` before imports
and measure a small thread limit. Not every NumPy/SciPy operation uses a threaded
backend, and multiplying worker count by core count is not a reliable estimate
of the actual number of running threads.

## PatientSampler: an exposure policy

`sampling="patient_balanced"` remains available. It selects a rotating set of
recording ranges, divides the epoch's requested windows approximately equally
among them, and interleaves requests within `patient_group_size` groups. Despite
the API's patient terminology, these ranges are recordings: several nights from
one subject can remain separate ranges. Uniform random windows instead weight
recordings in proportion to their candidate-window counts.

Use this mode when those exposure semantics are intended, not as an automatic
speed optimization. Earlier eight-worker pilots found little difference for
arousal on A30/4090 and an approximately 11% sleep improvement on 4090, while
A30 sleep was slightly slower. This does not establish a general throughput gain
or a predictive-quality effect.

Set `patients_per_epoch` and optionally `patient_group_size` when selecting this
mode. If omitted, group size is derived from the candidate budget, batch size,
workers and active recording count; it is a heuristic, not a measured optimum.
`BaseTrainer` advances the sampler epoch. Custom loops must call
`loader.sampler.set_epoch(epoch)` to advance the rotation.

Grouping does not assign dedicated files to individual workers, guarantee that
pages remain resident, or bound all prefetched work to one group. Rejection with
global fallback can leave the selected set. These semantics are unchanged.

## EDFCache was removed

The former `EDFCache` stored decoded **native-rate** channel arrays, not resampled
or filtered windows. It incurred whole-channel population, shared-registry and
mapping costs while leaving per-window preparation in place. It is no longer a
supported dataset or prediction feature.

In the completed A30 decision screen, four epochs of 50K candidates used group
size 8, capacity 32 in `/dev/shm`, and all combinations of 128/512 active
recordings and 2/8 workers for two workloads. The eight matched comparisons
showed one 2.8% total-time improvement, effectively tied sleep cases, and
slowdowns up to 219%. These single-seed measurements did not establish a reliable
benefit. The earlier capacity-4/group-8 configuration was substantially worse.

Single-recording packaged prediction diagnostics also did not justify keeping
it: sleep took 2.82 s without caching versus 4.69 s with capacity 3; arousal took
1.60 s versus 4.92 s. Each condition ran once, no-cache first, with eight workers;
cache creation/cleanup and prediction formatting were included, dataset
initialization excluded. This is limited evidence about the former implementation,
not proof that every form of caching is ineffective.

Migration: remove `EDFCache` imports, `dataset.set_edf_cache(...)`, the `cache=`
argument to `EDFFile.get_x/get_native`, and `edf_cache_patients` from
`PackagedModel.predict_dataset`. Prediction now uses direct window loading.
No replacement cache or prepared-window cache is installed automatically.

## NumpyDataset: an explicit fixed window pool

Materializing a loader pass remains available through
[`export_dataloader_to_numpy_dir`](../reference/datasets/NumpyDataset.md) and
`NumpyDataset`. This avoids repeated EDF decoding and dataset-side preprocessing
for the exported windows, but fixes their identities and dataset-side random
choices. Trainer-side augmentation can remain online.

Include export time when comparing total runtime. The native exporter assembles
arrays before writing shards: shard size does not bound peak export RAM. Check
available memory and disk, not just installed capacity. Signal payload alone is
`4 * windows * samples_per_window * channels` bytes for float32, before targets,
metadata, temporary arrays, loaders and model memory.

`in_memory=True` loads arrays into RAM; `in_memory=False` uses memory-mapped
arrays and lets the OS determine residency. Neither guarantees a speedup for
short runs. A fixed pool is an explicit training-data choice, not a transparent
replacement for fresh window sampling.

## Rejection and accepted work

Target rejection avoids decoding, but later quality checks and fallback aliases
can cost signal reads and processing. With the default BaseDataset retry budget,
`patient_then_global` can attempt up to 32 candidates per requested item. Report
accepted work rather than assuming the requested sample budget was delivered.

The loader's `drop_last=True` also rejects a batch if any item remains invalid;
it does more than omit the final short batch. This policy can discard otherwise
valid samples and must be considered for rejection-heavy datasets. Prediction
normally uses sequential access with rejection replacement disabled, so missing
windows remain visible in coverage rather than being replaced by other times.

## Measurement provenance

The 2026-10-25 profiling used Sleepwalker revision `c36cbcb`, Python 3.12.3,
NumPy 2.5.3, pandas 2.3.3, pyedflib 0.1.42/edflib 1.17 and Torch 2.14.1+cu130.
The accompanying system-paper workspace retains scripts and raw JSON/CSV in
`investigations/loading-performance/`, including `dataflow-inventory.md`, and
the A30 cache screen in `code/results-a30-shm/`. Historical cache runs require
the earlier implementation; they are not reproducible with a removed API simply
by rerunning old commands. These observations motivate the current defaults,
not hardware-independent guarantees.
