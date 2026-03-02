# extract_crops_from_mkv.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
from tqdm import tqdm


Point = Tuple[int, int]                 # (y, x)
Crop2Pts = Tuple[Point, Point]          # ((y1,x1),(y2,x2))


def find_mkv_files(root_dir: Path) -> List[Path]:
    return sorted(root_dir.rglob("*.mkv"))


def clamp_point(p: Point, w: int, h: int) -> Point:
    y, x = p
    y = max(0, min(y, h))
    x = max(0, min(x, w))
    return y, x


def clamp_crop(crop: Crop2Pts, w: int, h: int) -> Crop2Pts:
    (y1, x1), (y2, x2) = crop
    (y1, x1) = clamp_point((y1, x1), w, h)
    (y2, x2) = clamp_point((y2, x2), w, h)
    return (y1, x1), (y2, x2)


def is_valid_crop(crop: Crop2Pts) -> bool:
    (y1, x1), (y2, x2) = crop
    return (y2 > y1) and (x2 > x1)


def ensure_cam_dirs(parent_dir: Path, video_stem: str) -> List[Path]:
    out_dirs: List[Path] = []
    for cam_idx in range(1, 5):
        d = parent_dir / f"{video_stem}_cam{cam_idx}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs.append(d)
    return out_dirs


def process_video(
    video_path: Path,
    root_dir: Path,
    output_root: Path,
    frame_skip: int,
    crop_boxes: List[Crop2Pts],
    jpeg_quality: int = 95,
) -> None:
    if len(crop_boxes) != 4:
        raise ValueError("crop_boxes must have exactly 4 crops (cam1..cam4).")

    video_stem = video_path.stem

    # Mirror folder structure under OUTPUT_ROOT
    rel_parent = video_path.parent.relative_to(root_dir)
    mirrored_parent = output_root / rel_parent
    mirrored_parent.mkdir(parents=True, exist_ok=True)

    out_dirs = ensure_cam_dirs(mirrored_parent, video_stem)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        # Clamp crops if size known; otherwise clamp after first frame read
        clamped_crops: List[Crop2Pts] = []
        if w > 0 and h > 0:
            for c in crop_boxes:
                cc = clamp_crop(c, w, h)
                if not is_valid_crop(cc):
                    raise ValueError(f"Invalid crop after clamp for {video_path.name}: {c} -> {cc}")
                clamped_crops.append(cc)
        else:
            clamped_crops = list(crop_boxes)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_skip == 0:
                if w <= 0 or h <= 0:
                    h, w = frame.shape[:2]
                    clamped_crops = []
                    for c in crop_boxes:
                        cc = clamp_crop(c, w, h)
                        if not is_valid_crop(cc):
                            raise ValueError(f"Invalid crop after clamp for {video_path.name}: {c} -> {cc}")
                        clamped_crops.append(cc)

                for cam_i, ((y1, x1), (y2, x2)) in enumerate(clamped_crops, start=1):
                    crop = frame[y1:y2, x1:x2]  # OpenCV slicing: [rows, cols] => [y, x]
                    out_name = f"frame_crop_cam{cam_i}_{frame_idx:06d}.jpg"
                    out_path = out_dirs[cam_i - 1] / out_name
                    cv2.imwrite(str(out_path), crop, encode_params)

            frame_idx += 1

    finally:
        cap.release()


def main():
    # -----------------------------
    # CONFIG
    # -----------------------------
    ROOT_DIR = Path(r"../videos/")
    OUTPUT_ROOT = Path(r"../annotations/")
    FRAME_SKIP = 15  # 1 frame each X

    # FIXED crops from your screenshot: (left_top(y,x), bottom_right(y,x))
    CROP_BOXES: List[Crop2Pts] = [
        ((0,    0),    (720, 1280)),   # Camera 1
        ((0,    1320), (720, 2600)),   # Camera 2
        ((780,  0),    (1500, 1280)),  # Camera 3
        ((780,  1320), (1500, 2600)),  # Camera 4
    ]

    JPEG_QUALITY = 95
    # -----------------------------

    ROOT_DIR = ROOT_DIR.resolve()
    OUTPUT_ROOT = OUTPUT_ROOT.resolve()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    mkv_files = find_mkv_files(ROOT_DIR)
    if not mkv_files:
        print(f"No .mkv files found under: {ROOT_DIR}")
        return

    for video_path in tqdm(mkv_files, desc="Processing videos", unit="video"):
        try:
            process_video(
                video_path=video_path,
                root_dir=ROOT_DIR,
                output_root=OUTPUT_ROOT,
                frame_skip=FRAME_SKIP,
                crop_boxes=CROP_BOXES,
                jpeg_quality=JPEG_QUALITY,
            )
        except Exception as e:
            print(f"\n[ERROR] {video_path} -> {e}")

    print(f"\nDone. Output at: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()