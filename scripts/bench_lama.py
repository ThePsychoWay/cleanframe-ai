"""Micro-benchmark of LaMa inference variants on the machine it runs on (CPU or CUDA). Compares speed and the
difference from the plain FP32 result inside the hole. Usage: python scripts/bench_lama.py MODEL.pt [--size 300x200]"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def make_inputs(h: int, w: int, batch: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, (h // 8, w // 8, 3)).astype(np.uint8)
    import cv2
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    mask = np.zeros((1, 1, h, w), np.float32)
    mask[..., h // 3 : 2 * h // 3, w // 4 : 3 * w // 4] = 1.0
    return torch.from_numpy(img).repeat(batch, 1, 1, 1), torch.from_numpy(mask).repeat(batch, 1, 1, 1)


def bench(fn, image, mask, runs=8, warm=3):
    for _ in range(warm):
        fn(image, mask)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        out = fn(image, mask)
        if image.is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return float(np.median(times)), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--size", default="304x200")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    h, w = h - h % 8, w - w % 8
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(args.model, map_location=device).eval()
    image, mask = make_inputs(h, w, 1)
    image, mask = image.to(device), mask.to(device)
    hole = mask.bool().expand(-1, 3, -1, -1)
    print(f"device={device} threads={args.threads} crop={w}x{h}", flush=True)

    def fp32(i, m):
        with torch.no_grad():
            return model(i, m)

    t0, ref = bench(fp32, image, mask)
    print(f"fp32 TorchScript        {t0*1000:7.0f} ms/frame", flush=True)

    def diff(out):
        a = (out.float()[hole] * 255).clamp(0, 255)
        b = (ref.float()[hole] * 255).clamp(0, 255)
        return float((a - b).abs().mean())

    def bf16(i, m):
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16 if device == "cpu" else torch.float16):
            return model(i, m)
    try:
        t, out = bench(bf16, image, mask)
        print(f"autocast {'bf16' if device=='cpu' else 'fp16'}          {t*1000:7.0f} ms  x{t0/t:.2f}  mean|diff| in hole {diff(out):.2f}/255", flush=True)
    except Exception as e:  # noqa: BLE001
        print("autocast failed:", str(e)[:120])

    def cl(i, m):
        with torch.no_grad():
            return model(i.contiguous(memory_format=torch.channels_last), m.contiguous(memory_format=torch.channels_last))
    try:
        t, out = bench(cl, image, mask)
        print(f"channels_last fp32      {t*1000:7.0f} ms  x{t0/t:.2f}  mean|diff| {diff(out):.2f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print("channels_last failed:", str(e)[:120])

    try:
        frozen = torch.jit.optimize_for_inference(torch.jit.freeze(model))
        def fz(i, m):
            with torch.no_grad():
                return frozen(i, m)
        t, out = bench(fz, image, mask)
        print(f"jit freeze+optimize     {t*1000:7.0f} ms  x{t0/t:.2f}  mean|diff| {diff(out):.2f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print("freeze failed:", str(e)[:120])

    # smaller picture: shrink the crop, fill, and use the result only inside the hole
    for scale in (0.75, 0.5):
        hs, ws = int(h * scale) // 8 * 8, int(w * scale) // 8 * 8
        small_i = torch.nn.functional.interpolate(image, size=(hs, ws), mode="area")
        small_m = (torch.nn.functional.interpolate(mask, size=(hs, ws), mode="area") > 0).float()
        t, _ = bench(fp32, small_i, small_m)
        print(f"crop scaled to {scale:.2f}     {t*1000:7.0f} ms  x{t0/t:.2f}", flush=True)

    for batch in (2, 4):
        bi, bm = make_inputs(h, w, batch)
        bi, bm = bi.to(device), bm.to(device)
        t, _ = bench(fp32, bi, bm, runs=4, warm=2)
        print(f"batch {batch} fp32            {t*1000/batch:7.0f} ms/frame  x{t0/(t/batch):.2f}", flush=True)


if __name__ == "__main__":
    main()
