"""Adapter exposing a ``Featurizer`` as a model preprocessor: ``[B, T, C]`` windows in, ``[B, n_features]`` out.

It wraps the whole featurizer, so extractors run in parallel inside it while the preprocessor chain stays
in series. Featurization runs on the CPU in every forward pass, on ``n_jobs`` CPUs.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from sleepwalker.features.extractor import Extractor
from sleepwalker.features.featurizer import Featurizer
from sleepwalker.models.preprocessors.Preprocessor import Preprocessor
from sleepwalker.utils import logger


class Featurize(Preprocessor):
    """Turn ``[B, T, C]`` windows into ``[B, n_features]`` vectors using a ``Featurizer``.

    Args:
        featurizer: The configured :class:`Featurizer`, held by reference.
        n_jobs: CPUs that featurize each batch, stored with the package (``sleepwalker package-features
            --n-jobs``). ``None`` becomes 1, with a warning, at the first batch.
        fit_windows: Warmup windows to collect before fitting a stateful featurizer.

    There is no ``channels`` argument: selective preprocessing needs the shape preserved, and this changes
    it. Restrict channels on the ``Featurizer`` instead.
    """

    def __init__(
        self,
        featurizer: Featurizer,
        *,
        n_jobs: int | None = None,
        fit_windows: int = 1024,
    ) -> None:
        super().__init__(channels=None)  # never selective -- see the class docstring
        if fit_windows < 1:
            raise ValueError(f"fit_windows must be >= 1, got {fit_windows}.")

        self.featurizer = featurizer
        self.n_jobs = n_jobs
        self.fit_windows = int(fit_windows)
        self._buffer: list[np.ndarray] = []
        self._buffered = 0

    # ---- what this preprocessor emits ---- #
    @property
    def n_features(self) -> int:
        return self.featurizer.n_features

    @property
    def fitted(self) -> bool:
        """Whether this stage can transform. Read from the featurizer, so fitting it directly is seen too."""
        return self.featurizer.fitted

    # ---- the Preprocessor contract ---- #
    def requires_warmup(self) -> bool:
        """True when some extractor has state to fit, i.e. overrides ``Extractor.fit``."""
        return any(type(e).fit is not Extractor.fit for e in self.featurizer.extractors)

    def _update(self, data: torch.Tensor):
        """Buffer warmup windows, fitting the featurizer as soon as ``fit_windows`` is reached."""
        if self.featurizer.fitted:  # never refit, whoever fitted it
            return
        windows = data.detach().cpu().numpy()
        take = min(len(windows), self.fit_windows - self._buffered)
        if take > 0:
            self._buffer.append(windows[:take])
            self._buffered += take
        if self._buffered >= self.fit_windows:
            self.fit_from_buffer()

    def _transform(self, data: torch.Tensor) -> torch.Tensor:
        """Featurize one ``[B, T, C]`` batch into ``[B, n_features]``, first fitting on a partial buffer.

        With nothing buffered, a stateful featurizer raises rather than fitting on this batch.
        """
        self.fit_buffered()
        if self.n_jobs is None:
            logger.warning("Featurize: n_jobs is not set, so featurization runs on 1 CPU. Build the package with --n-jobs N to use N.")
            self.n_jobs = 1
        available = len(os.sched_getaffinity(0))
        if not 1 <= self.n_jobs <= available:
            raise ValueError(f"Featurize: n_jobs={self.n_jobs}, but this process may use 1 to {available} CPUs. Set n_jobs on the package's Featurize stage after loading it, see docs/how-to/features.md.")
        features = self.featurizer.transform_batch(data.detach().cpu().numpy(), n_jobs=self.n_jobs)
        # float32 keeps the feature matrix half the size of float64 through the cache and the head.
        return torch.from_numpy(np.asarray(features, dtype=np.float32)).to(data.device)

    def fit_buffered(self) -> None:
        """Fit on the buffered warmup windows if not fitted yet, e.g. when warmup ran out of data early."""
        if not self.featurizer.fitted and self._buffered > 0:
            self.fit_from_buffer()

    # ---- internals ---- #
    def fit_from_buffer(self) -> None:
        batch = np.concatenate(self._buffer, axis=0)
        self.featurizer.fit(batch)
        self._buffer = []
        logger.info(f"Featurize: fitted {len(self.featurizer.extractors)} extractor(s) on {len(batch)} windows.")
