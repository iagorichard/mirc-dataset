from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(root_dir: Path) -> List[Path]:
    return sorted([p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def ensure_mirrored_out_path(in_path: Path, in_root: Path, out_root: Path) -> Path:
    rel_parent = in_path.parent.relative_to(in_root)
    out_parent = out_root / rel_parent
    out_parent.mkdir(parents=True, exist_ok=True)
    return out_parent / in_path.name


def clamp_rect(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


def gaussian_blur_roi(img_bgr, x1: int, y1: int, x2: int, y2: int, ksize: int = 61) -> None:
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 3:
        ksize = 3
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return
    img_bgr[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (ksize, ksize), 0)


def pixelate_roi(img_bgr, x1: int, y1: int, x2: int, y2: int, blocks: int = 18) -> None:
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    bw = max(1, w // blocks)
    bh = max(1, h // blocks)
    small = cv2.resize(roi, (max(1, w // bw), max(1, h // bh)), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img_bgr[y1:y2, x1:x2] = pix


def main():
    # -----------------------------
    # CONFIG
    # -----------------------------
    IN_ROOT = Path(r"../annotations/not-blurred/")
    OUT_ROOT = Path(r"../annotations/blurred/")

    # YOLO weights (auto-download on first run)
    YOLO_WEIGHTS = "models/yolov8n.pt"   # lightweight; try "yolov8s.pt" for better recall
    IMGSZ = 960                   # higher helps small persons (try 1280 if needed)
    CONF = 0.25                   # lower -> more detections (0.15–0.3 typical)
    IOU = 0.5

    # Head region inside PERSON bbox (top fraction)
    HEAD_TOP_FRAC = 0.0           # start at top of bbox
    HEAD_HEIGHT_FRAC = 0.35       # blur top 35% of person bbox (tune 0.30–0.45)

    # Anonymization mode
    MODE = "pixelate"             # "blur" | "pixelate"
    BLUR_KSIZE = 61
    PIX_BLOCKS = 18

    # Fail-safe LGPD:
    # If no person detected => anonymize whole image (True is safest)
    ANON_WHOLE_IF_NONE = False
    # -----------------------------

    IN_ROOT = IN_ROOT.resolve()
    OUT_ROOT = OUT_ROOT.resolve()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    model = YOLO(YOLO_WEIGHTS)

    imgs = find_images(IN_ROOT)
    if not imgs:
        print(f"No images found under: {IN_ROOT}")
        return

    n_people_total = 0
    n_no_person = 0
    n_failed = 0

    for img_path in tqdm(imgs, desc="Anonymizing (YOLO person→head)", unit="img"):
        out_path = ensure_mirrored_out_path(img_path, IN_ROOT, OUT_ROOT)

        img = cv2.imread(str(img_path))
        if img is None:
            n_failed += 1
            continue

        h, w = img.shape[:2]

        try:
            # Run YOLO inference
            results = model.predict(
                source=img,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                verbose=False,
                classes=[0],  # COCO class 0 = person
            )

            boxes = []
            r0 = results[0]
            if r0.boxes is not None and len(r0.boxes) > 0:
                # xyxy in pixels
                for b in r0.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = [int(v) for v in b]
                    x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, w, h)
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))

            if not boxes:
                n_no_person += 1
                if ANON_WHOLE_IF_NONE:
                    if MODE == "blur":
                        img = cv2.GaussianBlur(img, (BLUR_KSIZE | 1, BLUR_KSIZE | 1), 0)
                    else:
                        pixelate_roi(img, 0, 0, w, h, blocks=PIX_BLOCKS)
            else:
                n_people_total += len(boxes)

                # Blur/pixelate head region of each person bbox
                for (x1, y1, x2, y2) in boxes:
                    pw = x2 - x1
                    ph = y2 - y1

                    hy1 = y1 + int(ph * HEAD_TOP_FRAC)
                    hy2 = y1 + int(ph * (HEAD_TOP_FRAC + HEAD_HEIGHT_FRAC))
                    hx1, hx2 = x1, x2

                    hx1, hy1, hx2, hy2 = clamp_rect(hx1, hy1, hx2, hy2, w, h)

                    if MODE == "blur":
                        gaussian_blur_roi(img, hx1, hy1, hx2, hy2, ksize=BLUR_KSIZE)
                    else:
                        pixelate_roi(img, hx1, hy1, hx2, hy2, blocks=PIX_BLOCKS)

            cv2.imwrite(str(out_path), img)

        except Exception:
            n_failed += 1

    print("\nDone.")
    print(f"Input root : {IN_ROOT}")
    print(f"Output root: {OUT_ROOT}")
    print(f"People detected (total): {n_people_total}")
    print(f"Images w/ no person: {n_no_person}  (anon_whole_if_none={ANON_WHOLE_IF_NONE})")
    print(f"Failed images: {n_failed}")


if __name__ == "__main__":
    main()