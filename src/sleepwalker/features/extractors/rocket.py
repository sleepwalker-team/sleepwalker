"""The rocket family of extractors -- random-convolution-kernel transforms behind the Extractor contract.

One shared base (:class:`RocketExtractorBase`) owns the windowing lifecycle every sktime rocket transform
needs -- centre-crop the ``[B, T, C]`` window to sktime's numpy3D ``[B, C, window_samples]``, refuse to
transform before ``fit``, and learn the output width up front for ``feature_names``. Each concrete
variant is a thin subclass that declares its own options and overrides :meth:`make_transform` (the one
place the variants differ) plus a ``feature_prefix`` so composed rockets never collide on feature names.

These are channel-**blind** (every channel is an equal series) and amplitude-agnostic (per-series
``normalise`` / internal normalisation).
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from sktime.transformations.panel.rocket import MiniRocketMultivariate, MultiRocketMultivariate, Rocket

from sleepwalker.features.context import ExtractionContext
from sleepwalker.features.extractor import Extractor
from sleepwalker.features.extractors import register_extractor


def to_array(out) -> np.ndarray:
    """sktime rocket transforms return a DataFrame; normalise to a 2D ndarray."""
    return out.to_numpy() if hasattr(out, "to_numpy") else np.asarray(out)


class RocketExtractorBase(Extractor):
    """Shared windowing lifecycle for the rocket variants; subclasses supply :meth:`make_transform`.

    A variant subclass declares its own kwargs (setting them on ``self`` **before** calling
    ``super().__init__``, because the base learns ``feature_names`` from a throwaway fit here), sets
    ``feature_prefix`` (its registry name, also the feature-column prefix), and implements
    :meth:`make_transform`. Everything else -- ``window_batch`` / ``transform_batch`` /
    ``transform_window`` / ``fit`` / ``feature_names`` -- is inherited.
    """

    #: Feature-column prefix and registry name; each concrete variant overrides it.
    feature_prefix: str = "rocket"

    def __init__(self, context: ExtractionContext, *, random_state: int = 0, **options) -> None:
        super().__init__(context, **options)
        if len(set(context.channel_roles)) != len(context.channel_roles):
            raise ValueError(f"channel_roles must be unique, got {list(context.channel_roles)}.")

        self.n_channels = len(context.channel_roles)
        self.window_samples = context.window_samples
        self.random_state = random_state

        self._transform = self.make_transform()
        self._fitted = False

        # Learn the output width now (config- and length-determined), so n_features / feature_names
        # satisfy the contract before the first real transform. A tiny throwaway fit reads it back.
        probe = self.make_transform()
        rng = np.random.default_rng(self.random_state)
        dummy = rng.standard_normal((4, self.n_channels, self.window_samples))
        width = int(to_array(probe.fit_transform(dummy)).shape[1])
        self._feature_names = [f"{self.feature_prefix}__{i:05d}" for i in range(width)]

    @abstractmethod
    def make_transform(self):
        """Build this variant's sktime transform."""

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def fitted(self) -> bool:
        return self._fitted

    def window_batch(self, x: np.ndarray) -> np.ndarray:
        """Centre-crop ``[B, T, C]`` to ``[B, C, window_samples]`` -- sktime's numpy3D layout."""
        if x.ndim != 3:
            raise ValueError(f"Expected a [B, T, C] batch, got shape {x.shape}.")
        _b, t, c = x.shape
        if c != self.n_channels:
            raise ValueError(
                f"Window has {c} channels but channel_roles describes {self.n_channels}: "
                f"{list(self.context.channel_roles)}."
            )
        if t < self.window_samples:
            raise ValueError(
                f"Window has {t} samples but this extractor needs at least {self.window_samples}."
            )
        start = (t - self.window_samples) // 2
        cropped = np.asarray(x[:, start : start + self.window_samples, :], dtype=np.float64)
        return np.ascontiguousarray(np.transpose(cropped, (0, 2, 1)))

    def transform_batch(self, x: np.ndarray, n_jobs: int = 1) -> np.ndarray:
        """Featurize a ``[B, T, C]`` batch into ``[B, n_features]`` (float32) on ``n_jobs`` threads. Raises until fitted."""
        if x.ndim != 3:
            raise ValueError(f"Expected a [B, T, C] batch, got shape {x.shape}.")
        if len(x) == 0:
            return np.empty((0, self.n_features), dtype=np.float32)
        self.require_fitted()
        self._transform.n_jobs = int(n_jobs)
        out = self._transform.transform(self.window_batch(x))
        return np.asarray(to_array(out), dtype=np.float32)

    def transform_window(self, x: np.ndarray) -> np.ndarray:
        """Featurize one ``[T, C]`` window. The intended entry point is :meth:`transform_batch`."""
        return self.transform_batch(x[None])[0]

    def require_fitted(self) -> None:
        """Refuse to transform unfitted rather than fit on whatever data arrived.

        The fitted state -- kernel dilations and, crucially, biases sampled from the data's own
        convolution quantiles -- is reused for every later transform. Fitting it implicitly on the first
        batch transformed would pick that batch silently: zeros from a torchinfo dummy pass, or validation
        and inference data. So fitting is always deliberate.
        """
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted. Its kernel biases come from the data it is fitted on, "
                "so fit it deliberately on training windows first: Featurizer.fit(batch), or preprocessor "
                "warmup (Featurize.update) before the first transform."
            )

    def fit(self, x: np.ndarray, y=None) -> "RocketExtractorBase":
        """Fit on a ``[B, T, C]`` batch (overrides the no-op). Required before any transform."""
        x3d = self.window_batch(x)
        self._transform.fit(x3d)
        self._fitted = True
        return self


@register_extractor("rocket")
class RocketExtractor(RocketExtractorBase):
    """ROCKET features over the fixed-length multivariate window (2 pooling features per kernel: PPV, max).

    Options:
        num_kernels: kernel count; width is ``num_kernels * 2``.
        normalise: per-series normalisation (kept ON so absolute amplitude is irrelevant).
        random_state: seeds the random kernels, making features reproducible.
    """

    feature_prefix = "rocket"

    def __init__(
        self,
        context: ExtractionContext,
        *,
        num_kernels: int = 10000,
        normalise: bool = True,
        random_state: int = 0,
        **options,
    ) -> None:
        self.num_kernels = int(num_kernels)
        self.normalise = bool(normalise)
        super().__init__(context, random_state=random_state, **options)

    def make_transform(self):
        return Rocket(
            num_kernels=self.num_kernels,
            normalise=self.normalise,
            random_state=self.random_state,
        )


@register_extractor("minirocket")
class MiniRocketExtractor(RocketExtractorBase):
    """MiniRocket features over the fixed-length multivariate window (PPV per kernel; internally normalised).

    Options:
        num_kernels: kernel count (sktime rounds it to a multiple of 84); width ~= ``num_kernels``.
        max_dilations_per_kernel: MiniRocket dilation cap (default 32).
        random_state: seeds the random kernels, making features reproducible.
    """

    feature_prefix = "minirocket"

    def __init__(
        self,
        context: ExtractionContext,
        *,
        num_kernels: int = 10000,
        max_dilations_per_kernel: int = 32,
        random_state: int = 0,
        **options,
    ) -> None:
        self.num_kernels = int(num_kernels)
        self.max_dilations_per_kernel = int(max_dilations_per_kernel)
        super().__init__(context, random_state=random_state, **options)

    def make_transform(self):
        return MiniRocketMultivariate(
            num_kernels=self.num_kernels,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            random_state=self.random_state,
        )


@register_extractor("multirocket")
class MultiRocketExtractor(RocketExtractorBase):
    """MultiRocket features over the fixed-length multivariate window.

    Options:
        num_kernels: kernel count; width is ``num_kernels * n_features_per_kernel * 2`` (default 6250 -> 50000).
        n_features_per_kernel: pooling features per kernel (MultiRocket default 4).
        normalise: per-series normalisation (kept ON so absolute amplitude is irrelevant).
        random_state: seeds the random kernels, making features reproducible.
    """

    feature_prefix = "multirocket"

    def __init__(
        self,
        context: ExtractionContext,
        *,
        num_kernels: int = 6250,
        n_features_per_kernel: int = 4,
        normalise: bool = True,
        random_state: int = 0,
        **options,
    ) -> None:
        self.num_kernels = int(num_kernels)
        self.n_features_per_kernel = int(n_features_per_kernel)
        self.normalise = bool(normalise)
        super().__init__(context, random_state=random_state, **options)

    def make_transform(self):
        return MultiRocketMultivariate(
            num_kernels=self.num_kernels,
            n_features_per_kernel=self.n_features_per_kernel,
            normalise=self.normalise,
            random_state=self.random_state,
        )
