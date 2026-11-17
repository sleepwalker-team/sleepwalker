# Model Interfaces

This page documents the model-layer interfaces that are currently visible in the repository. These are internal developer-facing conventions, not a stable public API.

## `BaseModel`

`sleepwalker.models.Basemodel.BaseModel` is the common base class for the repository's PyTorch models.

Observed expectations:

- inputs are tensors shaped `[B, T, C]`
- optional preprocessors run before feature extraction
- models expose a `features(x)` method
- models expose a `feature_dim()` method
- the full forward path is `classifier(features(x))`

In practice, the trainer and composite-model layers rely on this interface more than on any specific concrete architecture.

## Preprocessors

`BaseModel` can own a list of preprocessors. Current code assumes a preprocessor:

- is callable on a tensor
- may expose `requires_warmup()`
- may expose `update(x)`

Warmup is not performed by the model itself. Trainer code handles that.

## Why The Split Exists

The `features` / `classifier` split matters because:

- composite models reuse submodel embeddings without always reusing the final head
- deployment and test code often work with lightweight `BaseModel` subclasses
- trainer code only needs the full forward pass, but meta-model code needs direct access to embeddings

## Stability Notes

- The interface is consistent across current code and tests.
- It is still an internal convention rather than a versioned API surface.
- Shape assumptions are strong in practice, but not enforced through a formal protocol type.
