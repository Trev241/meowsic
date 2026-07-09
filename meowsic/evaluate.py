"""Objective quality metrics for a Meowsic render.

The goal of the deterministic pipeline is that the meow vocal tracks the original
lead vocal: it should re-trigger on the same note attacks (rhythm), follow the
same melodic ups and downs (pitch), and breathe with the same loudness dynamics
(envelope). These metrics quantify exactly that so parameter changes can be
compared objectively instead of by ear alone.

Everything here is numpy-only so it can run in the dependency-light core.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import detect_onsets, estimate_pitch_contour
from .types import AudioBuffer, MeowsicConfig


@dataclass(frozen=True)
class RenderQuality:
    """Objective comparison of a rendered meow vocal against the source vocal."""

    meow_score: float
    onset_f_measure: float
    onset_precision: float
    onset_recall: float
    coverage_ratio: float
    pitch_correlation: float
    envelope_correlation: float
    source_onsets: int
    render_onsets: int
    render_events: int
    details: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "meow_score": round(self.meow_score, 4),
            "onset_f_measure": round(self.onset_f_measure, 4),
            "onset_precision": round(self.onset_precision, 4),
            "onset_recall": round(self.onset_recall, 4),
            "coverage_ratio": round(self.coverage_ratio, 4),
            "pitch_correlation": round(self.pitch_correlation, 4),
            "envelope_correlation": round(self.envelope_correlation, 4),
            "source_onsets": self.source_onsets,
            "render_onsets": self.render_onsets,
            "render_events": self.render_events,
        }


def evaluate_render(
    source_vocal: AudioBuffer,
    rendered_meow: AudioBuffer,
    config: MeowsicConfig | None = None,
    *,
    onset_tolerance: float = 0.05,
    render_events: int = 0,
) -> RenderQuality:
    """Score how faithfully ``rendered_meow`` tracks ``source_vocal``.

    Returns a :class:`RenderQuality` whose ``meow_score`` in [0, 1] blends rhythm
    (onset alignment), melody (pitch-contour correlation), and dynamics (loudness
    envelope correlation). Higher is better.
    """

    config = config or MeowsicConfig()

    source_onsets = detect_onsets(source_vocal, config)
    render_onsets = detect_onsets(rendered_meow, config)
    precision, recall, f_measure = _onset_f_measure(source_onsets, render_onsets, onset_tolerance)
    coverage = float(render_onsets.size) / max(1, source_onsets.size)

    pitch_corr = _pitch_correlation(source_vocal, rendered_meow, config)
    env_corr = _envelope_correlation(source_vocal, rendered_meow, config)

    score = 0.5 * f_measure + 0.3 * max(0.0, pitch_corr) + 0.2 * max(0.0, env_corr)

    return RenderQuality(
        meow_score=float(score),
        onset_f_measure=float(f_measure),
        onset_precision=float(precision),
        onset_recall=float(recall),
        coverage_ratio=float(coverage),
        pitch_correlation=float(pitch_corr),
        envelope_correlation=float(env_corr),
        source_onsets=int(source_onsets.size),
        render_onsets=int(render_onsets.size),
        render_events=int(render_events),
    )


def _onset_f_measure(
    reference: np.ndarray,
    estimate: np.ndarray,
    tolerance: float,
) -> tuple[float, float, float]:
    """Greedy one-to-one onset matching within ``tolerance`` seconds."""

    if reference.size == 0 and estimate.size == 0:
        return 1.0, 1.0, 1.0
    if reference.size == 0 or estimate.size == 0:
        return 0.0, 0.0, 0.0

    ref = np.sort(reference)
    est = np.sort(estimate)
    used = np.zeros(est.size, dtype=bool)
    matches = 0
    for r in ref:
        candidates = np.where(~used & (np.abs(est - r) <= tolerance))[0]
        if candidates.size:
            nearest = candidates[np.argmin(np.abs(est[candidates] - r))]
            used[nearest] = True
            matches += 1

    precision = matches / est.size
    recall = matches / ref.size
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f_measure = 2 * precision * recall / (precision + recall)
    return precision, recall, f_measure


def _pitch_correlation(
    source_vocal: AudioBuffer,
    rendered_meow: AudioBuffer,
    config: MeowsicConfig,
) -> float:
    """Correlate the melodic shape (log-F0) of source and render over time."""

    source = estimate_pitch_contour(source_vocal, config)
    render = estimate_pitch_contour(rendered_meow, config)

    s_times, s_f0 = _voiced_series(source)
    r_times, r_f0 = _voiced_series(render)
    if s_times.size < 4 or r_times.size < 4:
        return 0.0

    # Sample both melodies on a shared time grid, keeping only frames where the
    # source is voiced and the render has nearby voiced content.
    grid = s_times
    r_on_grid = np.interp(grid, r_times, np.log2(r_f0), left=np.nan, right=np.nan)
    s_log = np.log2(s_f0)
    valid = ~np.isnan(r_on_grid)
    if valid.sum() < 4:
        return 0.0
    return _pearson(s_log[valid], r_on_grid[valid])


def _envelope_correlation(
    source_vocal: AudioBuffer,
    rendered_meow: AudioBuffer,
    config: MeowsicConfig,
) -> float:
    """Correlate the loudness (RMS) envelopes of source and render."""

    source = estimate_pitch_contour(source_vocal, config)
    render = estimate_pitch_contour(rendered_meow, config)
    if source.times.size < 4 or render.times.size < 4:
        return 0.0
    r_on_grid = np.interp(source.times, render.times, render.energy)
    return _pearson(source.energy, r_on_grid)


def _voiced_series(contour) -> tuple[np.ndarray, np.ndarray]:
    voiced = contour.voiced & (contour.f0_hz > 0)
    return contour.times[voiced], contour.f0_hz[voiced]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 1e-9:
        return 0.0
    return float(np.sum(a * b) / denom)
