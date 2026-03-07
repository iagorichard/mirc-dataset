import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))

import cv2
from pathlib import Path
from typing import List
from crop2pts import Crop2Pts
from files import ensure_cam_dirs
from maths import clamp_crop, is_valid_crop

def extract_frames_from_video(
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

def get_video_frames_quantity(video_file_path : Path) -> int:
    cap = cv2.VideoCapture(str(video_file_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames