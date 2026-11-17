import sys, time, httpx
from pathlib import Path
from urllib.parse import urlsplit
V = Path(sys.argv[1]); OUT = Path(sys.argv[2]); BASE = "http://localhost:8000"
def p(url): u = urlsplit(url); return f"{u.path}?{u.query}"
c = httpx.Client(base_url=BASE, timeout=60)
def region(i, x, y, w, h): return {"id": f"r{i}", "name": f"Mark {i}", "type": "rectangle", "keyframes": [{"frame": 0, "rect": {"x": x, "y": y, "w": w, "h": h}}]}
assert c.post("/api/auth/register", json={"email": "me3@example.com", "name": "Me", "password": "Clean-Frame-2026!"}).status_code in (200, 201)
data = V.read_bytes()
proj = c.post("/api/projects", json={"filename": "sample.mp4", "size_bytes": len(data)}).json()["project"]["id"]
urls = c.post(f"/api/projects/{proj}/upload/urls", json={"part_numbers": [1]}).json()["urls"]
etag = c.put(p(urls[0]["url"]), content=data).headers["ETag"]
c.post(f"/api/projects/{proj}/upload/complete", json={"parts": [{"part_number": 1, "etag": etag}]})
while c.get(f"/api/projects/{proj}").json()["status"] != "MASKING": time.sleep(1)
print("uploaded and analysed:", {k: c.get(f"/api/projects/{proj}").json().get(k) for k in ("width", "height", "frame_count", "fps")}, flush=True)
mask = {"version": 1, "regions": [region(1, .60, .064, .38, .082)]}
print("mask", c.put(f"/api/projects/{proj}/mask", json=mask).status_code)
print("render", c.post(f"/api/projects/{proj}/render", json={}).status_code, flush=True)
t = time.time(); last = None
while True:
    j = c.get(f"/api/projects/{proj}/render").json()["job"]
    key = (j["status"], j.get("stage"))
    if key != last: print(round(time.time() - t), key, flush=True); last = key
    if j["status"] in ("completed", "failed", "cancelled"): break
    time.sleep(3)
print("job:", j["status"], j.get("error")); print("warnings:", (j.get("result") or {}).get("warnings"))
if j["status"] == "completed":
    link = c.post(f"/api/projects/{proj}/export").json()
    OUT.write_bytes(c.get(p(link["url"])).content); print("saved", OUT, OUT.stat().st_size)
