"""Rebuilding the picture behind a mark from the other frames of the same video.

When the camera moves (a pan, a handheld shot, a zoom) or the mark itself moves, the background that one frame hides
is visible in other frames, at a different place on the screen. This engine finds those places, copies the real
pixels over, and takes the median of everything it found, so a person walking through the gap doesn't leave a ghost.
Nothing is invented and nothing is blurred: the pixels it writes were in your video.

What it cannot do, and says so: a mark that sits still on a still camera hides the same background in every frame, so
no frame has the missing pixels. Those pixels are handed to a second engine (LaMa, which generates a plausible
fill) if one is configured. If none is, the render stops with a message instead of smearing the area. The share of
the marked area rebuilt from real footage and the share generated is reported in the render's notes.

How a source frame is lined up with the target: the camera motion between neighbouring frames is estimated from the
pixels outside every mark (a similarity transform, from tracked corner points, robust to moving objects) and chained
across the window. A candidate is used only where the frame's own mark does not cover it, and only if, around the
mark, it matches the target frame closely enough to have been lined up correctly.

Limits: one global camera motion per frame pair, so strong parallax (near and far objects moving differently) leaves
small misalignments; a cut ends the window; the window of frames is held in memory (up to 33 frames at the highest quality).
"""

from __future__ import annotations

import logging
import zlib
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.ai.base import (
    DeviceInfo,
    Frame,
    InpaintingEngine,
    InpaintingError,
    InpaintingOptions,
    Mask,
    paired,
)
from app.ai.consistency import match_seam
from app.ai.image import ImageInpaintingEngine
from app.ai.roi import Roi, blend_into, prepare_mask

logger = logging.getLogger(__name__)

# Frames on each side of the target that may supply pixels.
WINDOW_FRAMES = {"fast": 6, "balanced": 10, "high": 16}
DEFAULT_MEMORY_MB = 1536  # the frames kept for the search window may use about this much
MIN_WINDOW_FRAMES = 3
MOTION_WIDTH = 960  # camera motion is measured on a copy this wide
STATIC_DIFFERENCE = 1.5  # mean grey-level difference below which two frames are the same picture
MIN_MOTION_POINTS = 12
CONTEXT_PX = 24  # real pixels this close to the mark are used to check that a source frame lines up
MIN_CONTEXT_PIXELS = 60
MAX_CONTEXT_ERROR = 10.0  # mean difference (0-255) on that ring above which a source frame is not trusted
MIN_SUPPORT = 2  # a pixel needs at least this many agreeing sources
# Reusing a generated fill: when the picture around a mark is the same as when the fill was generated, generating again would
# only produce the same patch (slowly), so the earlier one is used. "Same" = mean grey-level difference over the context at
# most this, with almost no pixel far off. Identical context (difference 0) is reused at every quality; the looser limits are
# a speed/quality trade the person chose with the quality setting.
FILL_REUSE_MEAN_DIFFERENCE = {"fast": 3.0, "balanced": 1.5, "high": 0.0}
FILL_REUSE_PIXEL_DIFFERENCE = 24  # a context pixel this far off counts as changed...
FILL_REUSE_MAX_CHANGED = 0.02  # ...and no more than this share of the context may have
MAX_CACHED_FILLS = 8
# Smoothing the generated pixels over time. A generated patch is a fresh guess on every frame, so on a moving picture it shimmers.
# Each frame's patch is blended with the previous frame's final patch (carried along with the picture's motion). The weight is 0
# at the patch's edge, where it must follow the real picture, and reaches SMOOTH_STRENGTH SMOOTH_EDGE_PX pixels inside. When the
# real pixels around the patch show that the previous frame no longer lines up, nothing is blended.
SMOOTH_STRENGTH = 0.75
SMOOTH_EDGE_PX = 6.0
SMOOTH_MAX_CONTEXT_ERROR = 10.0  # mean difference (0-255) on the real ring around the patch
SMOOTH_MIN_CONTEXT_PIXELS = 40
MAX_UNFILLED_WITHOUT_FALLBACK = 0.01  # share of a frame's marked pixels that may stay unrebuilt before the render stops

NO_SOURCE_MESSAGE = (
    "{percent:.0f}% of the marked area in frame {frame} is hidden in every nearby frame (a mark that doesn't move, on a "
    "camera that doesn't move), so there is nothing in this video to rebuild it from. Turn on the generative fill "
    "(on your computer: close the app and start it with `python run_local.py --generative-fill`; on a server: set "
    "AI_MODEL_PATH) or mark a smaller area."
)


@dataclass
class _Entry:
    index: int
    frame: Frame
    mask: Mask
    marked: bool
    usable: np.ndarray | None  # full-size uint8, 255 where this frame's real pixels may be copied; None = everywhere
    gray: np.ndarray  # small, for motion estimation
    scale: float
    exclude: np.ndarray  # small, 255 where a mark is (grown), ignored when measuring motion
    mask_key: int = 0  # identifies the marked pixels: two frames with the same key have the same mask
    step: np.ndarray | None = None  # 3x3: this frame -> the next (None = no usable relation: a cut)
    step_known: bool = False


@dataclass
class _Stats:
    marked_pixels: int = 0
    from_footage: int = 0
    generated: int = 0
    patched: int = 0
    frames_with_marks: int = 0
    frames_using_footage: int = 0
    generated_frames: set[int] = field(default_factory=set)
    fresh_fills: int = 0  # LaMa was run
    reused_fills: int = 0  # an earlier generated fill was used because the picture around the mark hadn't changed
    smoothed_frames: int = 0  # the generated patch was blended with the previous frame's


@dataclass
class _CachedFill:
    todo: np.ndarray  # crop-sized uint8: which pixels were generated
    context: np.ndarray  # crop-sized uint8 grey picture the fill was generated from
    generated: Frame  # crop-sized BGR


class TemporalReconstructionEngine(InpaintingEngine):
    provider = "reconstruct"
    name = "Reconstruction from other frames"
    is_development_only = False
    sends_frames_to_third_party = False

    def __init__(
        self,
        fallback: ImageInpaintingEngine | None = None,
        fallback_label: str = "",
        window: int | None = None,
        memory_mb: int = DEFAULT_MEMORY_MB,
    ) -> None:
        self._fallback = fallback
        self._fallback_label = fallback_label
        self._window = window
        self.memory_mb = memory_mb
        self._reduced_window: tuple[int, int] | None = None  # (frames wanted each side, frames kept each side) when memory cut it
        self._stats = _Stats()
        self._ones: np.ndarray | None = None
        self._fills: dict[tuple[int, int, int, int], _CachedFill] = {}
        self._previous: _Previous | None = None
        self._generated_map: np.ndarray | None = None  # set by _fill_the_rest for the frame being rebuilt

    # -- availability and description --------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._fallback is None or self._fallback.is_available()

    def unavailable_reason(self) -> str | None:
        return None if self._fallback is None else self._fallback.unavailable_reason()

    def device(self) -> DeviceInfo:
        if self._fallback is not None:
            fallback = self._fallback.device()
            if fallback is not None:
                return fallback
        return DeviceInfo(kind="cpu", name="CPU", note="Rebuilt from your own footage on the CPU; a long clip takes a while.")

    def expectation(self) -> str:
        base = "Marked areas are rebuilt from other frames of your video wherever the camera or the mark moves."
        if self._fallback is None:
            return base + " Areas hidden in every frame can't be rebuilt, and the render will say so."
        return base + f" Areas hidden in every frame are generated by {self._fallback_label or 'a second engine'}."

    def notes(self) -> list[str]:
        notes = self._core_notes()
        if notes and self._reduced_window is not None:
            wanted, kept = self._reduced_window
            notes.append(
                f"At this video size the frames searched for the missing picture were limited to {kept} each side of a frame "
                f"(instead of {wanted}) to stay within the server's memory budget."
            )
        return notes

    def _core_notes(self) -> list[str]:
        stats = self._stats
        if stats.marked_pixels == 0:
            return []
        footage = 100.0 * stats.from_footage / stats.marked_pixels
        generated = 100.0 * stats.generated / stats.marked_pixels
        if stats.generated == 0:
            message = f"{footage:.0f}% of the marked area was rebuilt from other frames of your video. Nothing was generated."
            if stats.patched:
                message += f" The last {100.0 * stats.patched / stats.marked_pixels:.1f}% (pixels no frame could supply) was patched from the pixels beside it."
            return [message]
        label = self._fallback_label or "the fallback engine"
        notes = [
            f"{footage:.0f}% of the marked area was rebuilt from other frames of your video; {generated:.0f}% was hidden in "
            f"every frame and was generated by {label}, which makes up plausible detail rather than recovering the real one."
        ]
        if stats.reused_fills:
            notes.append(
                f"The generated fill was made {stats.fresh_fills} times and reused {stats.reused_fills} times, on frames where the "
                "picture around the mark hadn't changed (the same patch each time, so it doesn't flicker)."
            )
        if stats.smoothed_frames:
            notes.append(
                f"On {stats.smoothed_frames} frames the generated patch was blended with the previous frame's (moved along with "
                "the picture) to reduce shimmer. That steadies the patch; it can't make it show what was really behind the mark."
            )
        return notes

    # -- processing --------------------------------------------------------------------------------------

    def process(self, frames: Iterable[Frame], masks: Iterable[Mask], options: InpaintingOptions) -> Iterator[Frame]:
        self.require_available()
        self.prepare()
        self._stats = _Stats()
        self._fills = {}
        self._previous = None
        reach = int(options.extra.get("window", self._window or WINDOW_FRAMES[options.quality]))
        self._reduced_window = None
        pairs = paired(frames, masks)
        first = next(pairs, None)
        if first is None:
            return
        reach = self._fit_window(reach, first[1])  # a 4K frame is 25 MB: the window shrinks to what memory allows
        pending = [first]
        buffer: deque[_Entry] = deque()
        exhausted = False
        target = 0
        try:
            while True:
                while not exhausted and (not buffer or buffer[-1].index < target + reach):
                    item = pending.pop() if pending else next(pairs, None)
                    if item is None:
                        exhausted = True
                        break
                    buffer.append(self._enter(item[0], item[1], item[2], options))
                while buffer and buffer[0].index < target - reach:
                    buffer.popleft()
                if not buffer or buffer[-1].index < target:
                    return
                current = buffer[target - buffer[0].index]
                yield self._rebuild(current, buffer, options)
                target += 1
        finally:
            self.close()

    def _fit_window(self, reach: int, frame: Frame) -> int:
        """Frames kept each side of the current one, cut down if the window would use more than the memory budget."""
        height, width = frame.shape[:2]
        per_frame = height * width * 5  # the picture, its mark and the "usable" map (the small grey copy is negligible)
        affordable = max(MIN_WINDOW_FRAMES, (self.memory_mb * 1024 * 1024 // per_frame - 1) // 2)
        if affordable < reach:
            self._reduced_window = (reach, affordable)
            logger.info("search window cut to fit memory", extra={"wanted": reach, "kept": affordable, "frame": f"{width}x{height}"})
            return affordable
        return reach

    def prepare(self) -> None:
        if self._fallback is not None:
            self._fallback.prepare()

    def close(self) -> None:
        if self._fallback is not None:
            self._fallback.close()

    # -- one frame in, one entry out -------------------------------------------------------------------------

    def _enter(self, index: int, frame: Frame, mask: Mask, options: InpaintingOptions) -> _Entry:
        height, width = frame.shape[:2]
        scale = min(1.0, MOTION_WIDTH / width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_size = (max(2, round(width * scale)), max(2, round(height * scale)))
        if scale < 1.0:
            gray = cv2.resize(gray, small_size, interpolation=cv2.INTER_AREA)
        binary = np.where(mask > 127, 255, 0).astype(np.uint8)
        marked = bool(binary.any())
        mask_key = zlib.crc32(binary.tobytes()) if marked else 0
        usable: np.ndarray | None = None
        exclude = np.zeros(gray.shape, np.uint8)
        if marked:
            radius = options.reach_px + 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            usable = cv2.bitwise_not(cv2.dilate(binary, kernel))
            exclude = cv2.dilate(cv2.resize(binary, small_size, interpolation=cv2.INTER_NEAREST), np.ones((7, 7), np.uint8))
        return _Entry(index=index, frame=frame, mask=mask, marked=marked, usable=usable, gray=gray, scale=scale, exclude=exclude, mask_key=mask_key)

    # -- camera motion --------------------------------------------------------------------------------------

    def _step(self, entries: list[_Entry], position: int) -> np.ndarray | None:
        """3x3 matrix taking pixels of entries[position] to entries[position + 1], or None across a cut."""
        entry = entries[position]
        if entry.step_known:
            return entry.step
        entry.step = _estimate_step(entry, entries[position + 1])
        entry.step_known = True
        return entry.step

    def _chains(self, entries: list[_Entry], target: int) -> dict[int, np.ndarray]:
        """Matrices taking every reachable neighbour's pixels into the target frame's coordinates."""
        chains: dict[int, np.ndarray] = {}
        running = np.eye(3)
        for position in range(target - 1, -1, -1):  # earlier frames: target-1, target-2, ...
            step = self._step(entries, position)
            if step is None:
                break
            running = running @ step
            chains[position] = running.copy()
        running = np.eye(3)
        for position in range(target + 1, len(entries)):  # later frames
            step = self._step(entries, position - 1)
            if step is None:
                break
            running = running @ np.linalg.inv(step)
            chains[position] = running.copy()
        return chains

    # -- rebuilding one frame --------------------------------------------------------------------------------

    def _rebuild(self, current: _Entry, buffer: deque[_Entry], options: InpaintingOptions) -> Frame:
        prepared = prepare_mask(current.mask, options)
        if prepared is None:
            self._previous = None
            return current.frame.copy()
        self._generated_map = None

        entries = list(buffer)
        position = entries.index(current)
        height, width = current.frame.shape[:2]
        pad = max(options.context_padding_px, options.feather_px + 1, CONTEXT_PX + 1)
        roi = prepared.bounds.expanded(pad, width, height)
        crop = np.ascontiguousarray(current.frame[roi.slices])
        hole = prepared.grown[roi.slices] > 0
        hole_count = int(hole.sum())
        self._stats.marked_pixels += hole_count
        self._stats.frames_with_marks += 1

        ring = cv2.dilate(hole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CONTEXT_PX + 1, 2 * CONTEXT_PX + 1))) > 0
        ring &= ~hole
        ring &= current.usable[roi.slices] > 0 if current.usable is not None else True

        colours: list[np.ndarray] = []  # each (hole_count, 3) float32
        supports: list[np.ndarray] = []  # each (hole_count,) bool
        hole_index = np.nonzero(hole)
        for neighbour_position, matrix in sorted(self._chains(entries, position).items(), key=lambda item: abs(item[0] - position)):
            neighbour = entries[neighbour_position]
            if neighbour.marked and neighbour.mask_key == current.mask_key and np.allclose(matrix, _IDENTITY, atol=1e-9):
                # Same picture position and the same mark: this frame hides exactly the pixels we are missing, so it can't
                # supply any of them (the same result as checking, without warping a frame to find out).
                continue
            warped, usable = self._warp(neighbour, matrix, roi, crop.shape[:2])
            context = ring & usable
            if int(context.sum()) < MIN_CONTEXT_PIXELS:
                continue  # can't tell whether this frame lines up: don't risk it
            error = float(np.abs(warped[context].astype(np.float32) - crop[context].astype(np.float32)).mean())
            if error > MAX_CONTEXT_ERROR:
                continue
            valid = usable[hole_index]
            if not valid.any():
                continue
            # Same scene points, so any average difference is a change of exposure between the frames: remove it.
            offset = (crop[context].astype(np.float32) - warped[context].astype(np.float32)).mean(axis=0)
            colours.append(warped[hole_index].astype(np.float32) + offset)
            supports.append(valid)

        filled = crop.copy()
        found = np.zeros(hole_count, bool)
        if supports:
            stack_valid = np.stack(supports)  # n, hole_count
            enough = stack_valid.sum(axis=0) >= MIN_SUPPORT
            if enough.any():
                stack = np.stack(colours)[:, enough, :]  # n, k, 3
                stack = np.where(stack_valid[:, enough, None], stack, np.nan)
                median = np.nanmedian(stack, axis=0)
                found[enough] = True
                painted = filled[hole_index]
                painted[enough] = np.clip(np.rint(median), 0, 255).astype(np.uint8)
                filled[hole_index] = painted
        self._stats.from_footage += int(found.sum())
        if found.any():
            self._stats.frames_using_footage += 1

        missing = hole_count - int(found.sum())
        if missing:
            filled = self._fill_the_rest(filled, hole, hole_index, found, missing, hole_count, current.index, options)

        output = current.frame.copy()
        blend_into(output, filled, prepared.alpha, roi)
        self._smooth_generated(output, roi, ring, current, entries, position, options)
        return output

    def _warp(self, neighbour: _Entry, matrix: np.ndarray, roi: Roi, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        shifted = matrix[:2].copy()
        shifted[:, 2] -= (roi.x0, roi.y0)
        dsize = (size[1], size[0])
        warped = cv2.warpAffine(neighbour.frame, shifted, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if neighbour.usable is None:
            if self._ones is None or self._ones.shape != neighbour.frame.shape[:2]:
                self._ones = np.full(neighbour.frame.shape[:2], 255, np.uint8)
            source = self._ones
        else:
            source = neighbour.usable
        usable = cv2.warpAffine(source, shifted, dsize, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return warped, usable > 0

    def _fill_the_rest(
        self,
        filled: Frame,
        hole: np.ndarray,
        hole_index: tuple[np.ndarray, np.ndarray],
        found: np.ndarray,
        missing: int,
        hole_count: int,
        frame_index: int,
        options: InpaintingOptions,
    ) -> Frame:
        if self._fallback is None:
            if missing / hole_count > MAX_UNFILLED_WITHOUT_FALLBACK:
                raise InpaintingError(NO_SOURCE_MESSAGE.format(percent=100.0 * missing / hole_count, frame=frame_index + 1))
            # A few stray pixels (the very edge of a mark at the picture's edge): leave the nearest rebuilt colour.
            todo = np.zeros(hole.shape, np.uint8)
            todo[hole_index[0][~found], hole_index[1][~found]] = 255
            self._stats.patched += missing
            return cv2.inpaint(filled, todo, 3, cv2.INPAINT_TELEA)
        todo = np.zeros(hole.shape, np.uint8)
        todo[hole_index[0][~found], hole_index[1][~found]] = 255
        result = filled.copy()
        for roi in _generation_boxes(todo, options):  # each mark on its own, with some picture around it as context
            crop, crop_todo = np.ascontiguousarray(filled[roi.slices]), np.ascontiguousarray(todo[roi.slices])
            generated = self._reused_fill(roi, crop, crop_todo, options)
            if generated is None:
                generated = self._fallback.inpaint_image(crop, crop_todo, options)
                self._remember_fill(roi, crop, crop_todo, generated)
                self._stats.fresh_fills += 1
            else:
                self._stats.reused_fills += 1
            region = result[roi.slices]
            region[crop_todo > 0] = generated[crop_todo > 0]
        self._stats.generated += missing
        self._stats.generated_frames.add(frame_index)
        self._generated_map = todo
        return match_seam(result, filled, todo)  # only the generated pixels are corrected, against the rebuilt ones

    # -- smoothing generated pixels over time ------------------------------------------------------------------

    def _smooth_generated(
        self, output: Frame, roi: Roi, ring: np.ndarray, current: _Entry, entries: list[_Entry], position: int, options: InpaintingOptions
    ) -> None:
        """Blend this frame's generated pixels with the previous frame's finished ones, then remember this frame's."""
        previous, self._previous = self._previous, None
        generated = self._generated_map
        self._generated_map = None
        crop = output[roi.slices]
        if generated is None or not generated.any():
            return  # nothing generated: real footage needs no smoothing, and the next frame has nothing to follow
        if options.temporal_smoothing and previous is not None and position > 0 and previous.index == current.index - 1:
            step = self._step(entries, position - 1)  # previous frame -> this one; None across a cut
            if step is not None and self._blend_previous(crop, roi, generated, ring, previous, step):
                self._stats.smoothed_frames += 1
        self._previous = _Previous(index=current.index, roi=roi, output=crop.copy(), generated=generated.copy())

    @staticmethod
    def _blend_previous(crop: Frame, roi: Roi, generated: np.ndarray, ring: np.ndarray, previous: _Previous, step: np.ndarray) -> bool:
        """True when the previous patch was blended in. `crop` is changed in place."""
        to_previous_crop = np.array([[1.0, 0.0, previous.roi.x0], [0.0, 1.0, previous.roi.y0], [0.0, 0.0, 1.0]])
        from_current_crop = np.array([[1.0, 0.0, -roi.x0], [0.0, 1.0, -roi.y0], [0.0, 0.0, 1.0]])
        matrix = (from_current_crop @ step @ to_previous_crop)[:2]
        size = (crop.shape[1], crop.shape[0])
        moved = cv2.warpAffine(previous.output, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        moved_generated = cv2.warpAffine(previous.generated, matrix, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 127
        inside = cv2.warpAffine(np.full(previous.output.shape[:2], 255, np.uint8), matrix, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
        context = ring & inside & ~moved_generated
        if int(context.sum()) < SMOOTH_MIN_CONTEXT_PIXELS:
            return False
        error = float(np.abs(moved[context].astype(np.float32) - crop[context].astype(np.float32)).mean())
        if error > SMOOTH_MAX_CONTEXT_ERROR:
            return False  # the picture around the patch changed (or the motion is off): last frame's patch no longer fits
        usable = (generated > 0) & moved_generated & inside
        if not usable.any():
            return False
        depth = cv2.distanceTransform((generated > 0).astype(np.uint8), cv2.DIST_L2, 3)
        weight = np.where(usable, SMOOTH_STRENGTH * np.clip(depth / SMOOTH_EDGE_PX, 0.0, 1.0), 0.0).astype(np.float32)
        active = weight > 0
        if not active.any():
            return False
        mixed = crop[active].astype(np.float32) * (1.0 - weight[active, None]) + moved[active].astype(np.float32) * weight[active, None]
        crop[active] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
        return True

    # -- reusing a generated fill ----------------------------------------------------------------------------

    def _reused_fill(self, roi: Roi, crop: Frame, crop_todo: Mask, options: InpaintingOptions) -> Frame | None:
        """The fill generated earlier for this same box, if the picture around the mark is still (nearly) the same."""
        cached = self._fills.get((roi.x0, roi.y0, roi.x1, roi.y1))
        if cached is None or not np.array_equal(cached.todo, crop_todo):
            return None
        outside = crop_todo == 0
        if not outside.any():
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        difference = np.abs(gray.astype(np.int16) - cached.context.astype(np.int16))[outside]
        if float(difference.mean()) > FILL_REUSE_MEAN_DIFFERENCE[options.quality]:
            return None
        if float((difference > FILL_REUSE_PIXEL_DIFFERENCE).mean()) > FILL_REUSE_MAX_CHANGED:
            return None
        return cached.generated

    def _remember_fill(self, roi: Roi, crop: Frame, crop_todo: Mask, generated: Frame) -> None:
        if len(self._fills) >= MAX_CACHED_FILLS:
            self._fills.pop(next(iter(self._fills)))  # the oldest
        self._fills[(roi.x0, roi.y0, roi.x1, roi.y1)] = _CachedFill(
            todo=crop_todo.copy(), context=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), generated=generated.copy()
        )


@dataclass
class _Previous:
    """The last frame's finished patch area, kept to damp the next frame's generated pixels."""

    index: int
    roi: Roi
    output: Frame  # roi-sized crop of the finished frame
    generated: np.ndarray  # roi-sized uint8, 255 where the pixels were generated


_IDENTITY = np.eye(3)
GENERATION_CONTEXT_PX = 48  # picture kept around a mark for the generative fill to look at


def _generation_boxes(todo: np.ndarray, options: InpaintingOptions) -> list[Roi]:
    """Boxes (marked pixels plus context) for the generative fill. Marks close together share a box; far apart ones don't,
    so the model never sees a whole frame to fill a few small marks (which is slower and shrinks the picture first)."""
    context = max(options.context_padding_px, GENERATION_CONTEXT_PX)
    near = cv2.dilate(todo, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * context + 1, 2 * context + 1)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(near, connectivity=8)
    boxes = []
    for x, y, w, h in stats[1:, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]:
        boxes.append(Roi(int(x), int(y), int(x + w), int(y + h)))  # already includes the context: it was grown first
    return boxes


def _estimate_step(first: _Entry, second: _Entry) -> np.ndarray | None:
    """Similarity transform (as 3x3, full-resolution pixels) taking `first`'s picture to `second`'s, measured on the
    pixels outside every mark. Identity for a picture that hasn't moved, None for a cut or when nothing can be followed."""
    if first.gray.shape != second.gray.shape:
        return None
    allowed = cv2.bitwise_not(cv2.max(first.exclude, second.exclude))
    if not allowed.any():
        return None
    difference = float(cv2.absdiff(first.gray, second.gray)[allowed > 0].mean())
    if difference < STATIC_DIFFERENCE:
        return np.eye(3)
    points = cv2.goodFeaturesToTrack(first.gray, maxCorners=400, qualityLevel=0.01, minDistance=6, mask=allowed)
    if points is not None and len(points) >= MIN_MOTION_POINTS:
        moved, status, _ = cv2.calcOpticalFlowPyrLK(first.gray, second.gray, points, None, winSize=(21, 21), maxLevel=4)
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(second.gray, first.gray, moved, None, winSize=(21, 21), maxLevel=4)
        good = (status.ravel() == 1) & (back_status.ravel() == 1) & (np.linalg.norm(back - points, axis=2).ravel() < 0.8)
        if int(good.sum()) >= MIN_MOTION_POINTS:
            matrix, inliers = cv2.estimateAffinePartial2D(points[good], moved[good], method=cv2.RANSAC, ransacReprojThreshold=0.8)
            if matrix is not None and inliers is not None and int(inliers.sum()) >= MIN_MOTION_POINTS:
                full = np.vstack([matrix, [0.0, 0.0, 1.0]])
                if first.scale != 1.0:  # small -> full resolution: x_full = x_small / scale
                    to_small = np.diag([first.scale, first.scale, 1.0])
                    full = np.linalg.inv(to_small) @ full @ to_small
                return full
    return None
