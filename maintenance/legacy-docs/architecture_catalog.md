# Architecture Catalog

This page collects the main model backbones currently present in the
repository.

The codebase does not define a stable public model API beyond the shared
`BaseModel` contract described in
[model_interfaces.md](/root/projects/sleepwalker/docs/model_interfaces.md).
The list below is therefore an inventory of current implementations, not a
guarantee of long-term support.

## Main single-model backbones

- `SleepTransformer`
- `SeqSleepNet`
- `AttnSleep`
- `MRASleepNet`
- `USleep`
- `UTime`
- `TinySleepNet`

Confirmed from the training scripts:

- `train_sleep.py` switches between several sleep-staging backbones including
  `TinySleepNet`
- `train_arousal.py`, `train_desaturation.py`, `train_lm.py`,
  `train_body_position.py`, and `train_noisy.py` use `UTime` in current
  workflows
- `train_multilabel.py` composes multiple backbones into a multitask model

## Composition layers

- `MultiModel` fuses embeddings from multiple submodels into one shared head
- `MetaModel` reuses fused embeddings but exposes one head per configured task

These composition layers are documented in
[multi_modeling.md](/root/projects/sleepwalker/docs/multi_modeling.md).

## Current modeling caveats

- several architecture files reference papers, but some explicitly note that
  the implementation is only an approximate or generalized port
- output shapes vary between pooled classification and sequence-style outputs
- preprocessor usage is model-dependent rather than globally uniform
- most architectures are intended for internal ML experimentation, not as
  standalone reusable packages

## Related pages

- [model_architectures.md](/root/projects/sleepwalker/docs/model_architectures.md)
- [model_interfaces.md](/root/projects/sleepwalker/docs/model_interfaces.md)
- [preprocessors.md](/root/projects/sleepwalker/docs/preprocessors.md)
- [multi_modeling.md](/root/projects/sleepwalker/docs/multi_modeling.md)
