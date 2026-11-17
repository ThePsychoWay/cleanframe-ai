"""Measure shimmer in generated fill: run the production reconstruction engine on a slice of a clip and report how much the
filled area changes from frame to frame, next to how much the original footage changes in the same place.

    AI_MODEL_PATH=/path/to/big-lama.pt python scripts/bench_fill.py clip.mp4 marks.json START COUNT out.npz [smooth=1|0]

`marks.json` is a mask document (what the editor saves, or `document_for(...).model_dump_json()` from the detector). Prints the
engine's own notes, the render's artifact-monitor fraction, the QC flicker ratio, and the mean per-frame change inside the mask
for the result and for the original. `out.npz` keeps result/mask/original frames for side-by-side crops.
"""
import json
import os
import sys
import time
from pathlib import Path

if not os.environ.get("AI_MODEL_PATH"):
    sys.exit("Set AI_MODEL_PATH to a LaMa model (big-lama.pt).")
os.environ.setdefault("AI_PROVIDER", "reconstruct")
os.environ.setdefault("RECONSTRUCT_FALLBACK", "lama")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import cv2
import numpy as np
from app.ai.registry import create_engine
from app.ai.base import InpaintingOptions
from app.schemas.mask import MaskDocument
from app.video.masks import MaskSequence
from app.video.verify import ArtifactMonitor
from app.services.quality import _flicker
clip,marks,start,count,out=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
smooth=bool(int(sys.argv[6])) if len(sys.argv)>6 else True
cap=cv2.VideoCapture(clip); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W=int(cap.get(3)); H=int(cap.get(4))
doc=MaskDocument.model_validate(json.load(open(marks)))
seq=MaskSequence(doc,None,width=W,height=H,frame_count=total)
cap.set(cv2.CAP_PROP_POS_FRAMES,start); frames=[]
for _ in range(count):
    ok,f=cap.read(); assert ok; frames.append(f)
masks=[(seq.mask(start+i)>0).astype(np.uint8)*255 for i in range(count)]
eng=create_engine(); eng.prepare()
opts=InpaintingOptions(temporal_smoothing=smooth)
t=time.time(); res=list(eng.process(iter(frames),iter(masks),opts)); dt=time.time()-t
print("render s",round(dt,1),"per frame",round(dt/count,2)); [print(" note:",n) for n in eng.notes()]
mon=ArtifactMonitor()
for r,m in zip(res,masks): mon.update(r,m)
grey=lambda L:np.stack([cv2.cvtColor(x,cv2.COLOR_BGR2GRAY).astype(np.float32) for x in L])
stack=np.stack([m>0 for m in masks])
fl_out=_flicker(grey(res),stack); fl_in=_flicker(grey(frames),stack)
union=stack.any(axis=0)
def inside_change(L):
    g=grey(L); d=[]
    for i in range(1,len(L)):
        b=stack[i]&stack[i-1]
        if b.any(): d.append(np.abs(g[i]-g[i-1])[b].mean())
    return float(np.mean(d))
print(json.dumps({"artifact_fraction":round(mon.fraction,3),"flagged":mon.flagged,"evaluated":mon.evaluated,"quality_flicker_ratio_out":None if fl_out is None else round(fl_out,2),"ratio_original_footage":None if fl_in is None else round(fl_in,2),"mean_change_inside_out":round(inside_change(res),2),"mean_change_inside_original":round(inside_change(frames),2)}))
np.savez_compressed(out,res=np.stack(res),mask=np.stack(masks),orig=np.stack(frames))
