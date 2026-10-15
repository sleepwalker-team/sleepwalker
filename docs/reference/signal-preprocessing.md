# Recording scaling and delayed channel slices

These reusable components provide recording scaling, selected channel scaling/clipping and delayed window slicing. Their motivating application is DRIVEN; the branch model, association recipe, batch trainer and native training/evaluation configurations remain pending.

## Recording scaling

`sleepwalker.datasets.normalizer.RecordingNormalizer.RecordingMeanTrimmedZScore` uses the existing recording-fitted channel processor interface. It fits the full-recording mean, retains the samples closest to that mean, and learns their mean and **sample** standard deviation (`ddof=1`). Windows use these fitted statistics unchanged; output units are `dimensionless`.

`retained_fraction` defaults to `0.95`, must be finite and in `(0, 1]`, and retains exactly `floor(retained_fraction * N)` samples. Cutoff ties are resolved in recording order. The accepted input shapes are `[N]` and `[N,1]` for one logical source/physical alias. Fewer than two retained samples, non-finite recordings or zero variance reject the recording (`fit` returns `None`). A rejected refit clears earlier fitted statistics. Calling before a successful fit or fitting a window raises an error. The dataset already clones fitted state independently per recording/source/alias.

## Delayed signal slices

`sleepwalker.models.preprocessors.DelayedChannelSlices.DelayedChannelSlices` accepts floating-point `[B,T,C]` tensors with arbitrary channel count/order. Configure `sampling_frequency` in Hz, `window_duration` in seconds and `slices` as `(input_channel, delay_seconds)` pairs. Each pair becomes one output channel; pair order determines output order. Channels, delays and entire pairs may repeat. This step performs no scaling, clipping or physiological interpretation.

Output is `[B,window_duration * sampling_frequency,len(slices)]`. Rates/durations/delays must give integral sample indices. At least `(window_duration + max(delay_seconds)) * sampling_frequency` samples are required; extra context is allowed and missing future context is rejected. Delays must be finite and non-negative; channel indices must be non-negative integers present in the input. An empty slice list, non-finite input or integer tensor is rejected. Input is unchanged; dtype/device are preserved. `requires_warmup()` returns false and `update()` learns no statistics.

## Selected channel scaling and clipping

`sleepwalker.models.preprocessors.ScaleChannels.ScaleChannels` configures fixed multipliers through `scales: {input_channel: multiplier}`. Optional `clip: {input_channel: [minimum, maximum]}` bounds are applied after scaling. Unspecified channels retain their values. Scaling and clipping can target different channels; clipping is disabled by default. Multipliers and bounds must be finite, bounds ordered, and indices present in the input. This step preserves input shape/dtype/device, leaves input unchanged and needs no warmup.

Both mappings use channel indices in this step's input. Placing scaling before slicing scales each selected source channel and all its later delayed copies. Placing it after slicing scales selected output copies independently. Scaling never infers channel units or semantics.

## DRIVEN example and source

The reusable slice operation was motivated by [DRIVEN's pinned data-preparation source](https://github.com/LCSB-SCG/DRIVEN/blob/dce8bfadd05d249ca7b55fa5d2d1974b52ba35ec/create_hdf5_from_psg/gen_img_complete.m). Its ABD/SpO2 recipe explicitly chooses the input order and scales physical-percent SpO2 to a fraction before slicing:

```yaml
preprocessors:
  - name: sleepwalker.models.preprocessors.ScaleChannels.ScaleChannels
    scales: {1: 0.01}
    clip: {1: [0, 1]}
  - name: sleepwalker.models.preprocessors.DelayedChannelSlices.DelayedChannelSlices
    sampling_frequency: 64
    window_duration: 30
    slices: [[0, 0], [1, 0], [1, 10], [1, 15], [1, 20], [1, 25]]
```

Here input is `[B,3520,2]` in recording-scaled ABD, physical-percent SpO2 order; output is `[B,1920,6]`. These semantics belong to the recipe. Forward delays require future data, making this an offline window benchmark with latency of at least the largest delay. A model with `preprocessors=[]` must declare its already sliced input geometry separately through `input_spec()`.
