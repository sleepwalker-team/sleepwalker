# Channel Routing

This page explains the current channel-routing logic used across datasets and composite models.

## Dataset-Level Grouping

At the dataset level, `ChannelConfig` can declare a `group`.

Observed behavior in `BaseDataset`:

- channels with the same group are treated as alternatives for one conceptual input
- one available channel is sampled per group when building a sample
- the selected column is renamed to the group name before it reaches downstream code

This means the model can see stable conceptual input names such as `EEG` even when the underlying EDF channel may vary by patient or by sampled choice.

## Quality Channels

`ChannelConfig` can also declare a `quality_name`.

When the selected signal has a quality channel:

- the quality channel is loaded too
- it is passed into `prepare_sample` as `quality_data`
- it does not become part of the main model input unless user code explicitly uses it

## Model-Level Routing

At the model level, `MetaModelEntry` maps channel names to submodels.

Observed behavior in `MultiModel` and `MetaModel`:

- the composite model expects a global channel order
- each submodel receives only the channel indices mapped to its entry
- fused embeddings are concatenated after each submodel processes its slice

## Why This Matters

The repository uses routing in two distinct ways:

- dataset grouping smooths over dataset-specific channel availability
- model routing lets different submodels specialize on different signal subsets

These layers are complementary. Grouping happens before the tensor reaches the model; routing happens inside the composite model once a stable channel layout already exists.

## Current Caveats

- Routing is name-based and convention-driven.
- There is no separate schema or validation layer beyond the current constructor checks.
- Training scripts are still the clearest source for intended channel layouts in real experiments.
