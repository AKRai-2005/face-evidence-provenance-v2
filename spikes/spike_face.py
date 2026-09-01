"""Phase 0 / Gate 0 spike: verify InsightFace buffalo_l works on THIS machine and
that two *different* photographs of the same public figure yield a high cosine.

Embeds EVERY detected face per image (not just the largest) and prints the full
cross-image matrix. Rationale: group photos defeat a 'largest face' heuristic --
the real pipeline must take max-over-faces per candidate.
"""
import hashlib, pathlib, time
import numpy as np, requests, cv2
from insightface.app import FaceAnalysis

UA = {"User-Agent": "HHGoa2026-Task3-Research/0.1 (ashutoshkumarrai19o7@gmail.com) python-requests"}
FIX = pathlib.Path("data/fixtures"); FIX.mkdir(parents=True, exist_ok=True)
C = "https://upload.wikimedia.org/wikipedia/commons"

IMAGES = {
    # subject A -- Sundar Pichai, three separate events/years
    "A1_pichai_hanoi2015.jpg":  f"{C}/thumb/a/a5/Meet_Google_CEO_Sundar_Pichai_%40Hanoi%2C_Vietnam_%2823291242774%29.jpg/1280px-Meet_Google_CEO_Sundar_Pichai_%40Hanoi%2C_Vietnam_%2823291242774%29.jpg",
    "A2_pichai_warsaw2022.jpg": f"{C}/thumb/8/8f/Mateusz_Morawiecki_spotka%C5%82_si%C4%99_z_CEO_Google_Sundar_Pichai_w_KPRM_%282022.03.29%29_01.jpg/1280px-Mateusz_Morawiecki_spotka%C5%82_si%C4%99_z_CEO_Google_Sundar_Pichai_w_KPRM_%282022.03.29%29_01.jpg",
    "A3_pichai_warsaw_crop.jpg":f"{C}/7/7b/Mateusz_Morawiecki_spotka%C5%82_si%C4%99_z_CEO_Google_Sundar_Pichai_w_KPRM_%282022.03.29%29_01_%28cropped%29.jpg",
    # subject B -- Satya Nadella, two separate photos (distinct identity control)
    "B1_nadella_2017.jpg":      f"{C}/thumb/4/44/MS-Exec-Nadella-Satya-2017-08-31-22-2.jpg/1280px-MS-Exec-Nadella-Satya-2017-08-31-22-2.jpg",
    "B2_nadella_smiling.jpg":   f"{C}/thumb/1/19/Satya_smiling-print.jpg/1280px-Satya_smiling-print.jpg",
}

def fetch(name, url):
    p = FIX / name
    if not p.exists():
        last = None
        for attempt in range(4):
            r = requests.get(url, headers=UA, timeout=45)
            if r.status_code == 200:
                p.write_bytes(r.content); break
            last = f"{r.status_code} {r.reason}"; time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"giving up after retries: {last}")
        time.sleep(1.0)
    return p, hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    faces_by_img = {}   # name -> list of (idx, det_score, bbox, emb)
    for name, url in IMAGES.items():
        try:
            p, sha = fetch(name, url)
        except Exception as e:
            print(f"  DOWNLOAD FAILED {name}: {e}"); continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"  DECODE FAILED {name}"); continue
        fs = sorted(app.get(img), key=lambda f: -f.det_score)
        if not fs:
            print(f"  NO FACE {name}"); continue
        faces_by_img[name] = [(i, float(f.det_score),
                               [int(v) for v in f.bbox], f.normed_embedding)
                              for i, f in enumerate(fs)]
        t0 = time.time(); app.get(img); dt = time.time() - t0
        print(f"  {name:26s} {img.shape[1]}x{img.shape[0]} sha={sha[:10]}.. "
              f"faces={len(fs)} det={[round(x[1],3) for x in faces_by_img[name]]} {dt:.2f}s/img")

    print("\n  --- max cosine over all face pairs, per image pair ---")
    names = list(faces_by_img)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            best = max(((float(np.dot(ea, eb)), ia, ib)
                        for ia, _, _, ea in faces_by_img[a]
                        for ib, _, _, eb in faces_by_img[b]))
            cos, ia, ib = best
            same = "SAME" if a[0] == b[0] else "DIFF"
            print(f"  [{same}] {a:26s} f{ia} vs {b:26s} f{ib}  maxcos={cos:+.4f}")

if __name__ == "__main__":
    main()
