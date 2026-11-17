"""Plain data the detector passes between its steps. Coordinates are survey pixels unless a name says otherwise."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


@dataclass
class Candidate:
    box: Box
    score: float  # mean evidence inside: how strongly a structure stays put while the picture changes
    activity: float  # how much the picture moves around it (grey levels per frame)
    window: tuple[int, int]  # sample range the evidence came from
    template: np.ndarray  # float32, the overlay's own look (window mean of the high-pass picture)
    support: np.ndarray  # uint8, 1 where the template holds overlay pixels
    texty: bool  # many small strokes in a row
    coherence: float = 0.0  # median over the structure of |mean|/mean|.|: 1 = the same on every frame
    basis: str = "motion"  # "motion": stays put while the picture moves; "appearance": looks like a mark and sits where marks sit


@dataclass
class Overlay:
    id: int
    kind: str  # "text" | "logo"
    template: np.ndarray
    support: np.ndarray
    score: float
    activity: float
    present: np.ndarray  # bool per sample
    x: np.ndarray  # top-left of the template box per sample (survey px), float32
    y: np.ndarray
    s: np.ndarray  # scale relative to the template
    match: np.ndarray  # match score per sample (0 where absent)
    coherence: float = 0.0  # over the whole track, see service.track_stats
    strength: float = 0.0
    verdict: str = "remove"  # "remove" | "skipped"
    reason: str = ""
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    basis: str = "motion"
    shapes: dict[int, np.ndarray] | None = None  # sample -> 4 corner points (survey px), for marks measured shape by shape

    @property
    def frames_present(self) -> int:
        return int(self.present.sum())

    def box_at(self, i: int) -> tuple[float, float, float, float]:
        h, w = self.template.shape
        return float(self.x[i]), float(self.y[i]), w * float(self.s[i]), h * float(self.s[i])


@dataclass
class DetectionReport:
    overlays: list[Overlay]
    timings: dict[str, float]
    notes: list[str]
    samples: int
    stride: int
    picture_activity: float  # median grey-level change per frame across the picture
