"""Automatic overlay detection: find what is stuck on top of a video, on every frame, without anyone marking it.

Pipeline (each step is in its own module and works on a small grey copy of the whole timeline):

    survey.py      decode every frame once at low resolution
    evidence.py    where does a structure stay put while the picture under it changes?  (static, transparent, text)
    candidates.py  turn evidence into boxes, each with a template of what the overlay looks like
    follow.py      find each template on every frame (moves, jumps, resizes, appears, disappears, repeats elsewhere)
    build.py       turn the per-frame findings into a mask document the renderer already understands
    service.py     the whole thing, with a per-step time record and an honest report
"""
