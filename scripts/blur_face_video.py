from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
from tqdm import tqdm
from ultralytics import YOLO

import numpy as np
import torch
import head_segmentation.segmentation_pipeline as head_seg


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

def build_person_mask_from_result(result, det_idx: int, h: int, w: int) -> np.ndarray | None:
    """
    Builds a binary mask for one detected person instance using result.masks.xy.
    Returns:
        mask: uint8 array with shape (h, w), values {0, 255}
        or None if masks are unavailable.
    """
    if result.masks is None:
        return None

    # Use polygon representation because it is already in image coordinates.
    if det_idx >= len(result.masks.xy):
        return None

    poly = result.masks.xy[det_idx]
    if poly is None or len(poly) == 0:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32)

    # fillPoly expects shape (N, 1, 2) or list of polygons
    cv2.fillPoly(mask, [pts], 255)
    return mask


def make_blurred_copy(img_bgr, ksize: int = 61):
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 3:
        ksize = 3
    return cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)


def make_pixelated_copy(img_bgr, blocks: int = 18):
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return img_bgr.copy()

    bw = max(1, w // blocks)
    bh = max(1, h // blocks)

    small = cv2.resize(
        img_bgr,
        (max(1, w // bw), max(1, h // bh)),
        interpolation=cv2.INTER_LINEAR
    )
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pix


def apply_effect_limited_by_mask(
    img_bgr,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    limiter_mask: np.ndarray | None,
    mode: str = "pixelate",
    blur_ksize: int = 61,
    pix_blocks: int = 18,
) -> None:
    """
    Applies blur/pixelation only inside:
        ROI rectangle ∩ limiter_mask

    If limiter_mask is None or empty in the ROI, fallback to the old rectangular behavior.
    """
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return

    # Fallback: preserve the old behavior if there is no usable mask
    if limiter_mask is None:
        if mode == "blur":
            gaussian_blur_roi(img_bgr, x1, y1, x2, y2, ksize=blur_ksize)
        else:
            pixelate_roi(img_bgr, x1, y1, x2, y2, blocks=pix_blocks)
        return

    mask_roi = limiter_mask[y1:y2, x1:x2]
    if mask_roi.size == 0 or not np.any(mask_roi):
        # Fallback to original behavior if the mask does not cover the ROI
        if mode == "blur":
            gaussian_blur_roi(img_bgr, x1, y1, x2, y2, ksize=blur_ksize)
        else:
            pixelate_roi(img_bgr, x1, y1, x2, y2, blocks=pix_blocks)
        return

    if mode == "blur":
        processed = make_blurred_copy(roi, ksize=blur_ksize)
    else:
        processed = make_pixelated_copy(roi, blocks=pix_blocks)

    # Apply only where the person mask exists inside the ROI
    binary = mask_roi > 0
    roi[binary] = processed[binary]

def find_videos(root_dir: Path) -> List[Path]:
    return sorted([p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def ensure_mirrored_out_path(in_path: Path, in_root: Path, out_root: Path, out_suffix: str = ".mp4") -> Path:
    rel_parent = in_path.parent.relative_to(in_root)
    out_parent = out_root / rel_parent
    out_parent.mkdir(parents=True, exist_ok=True)
    return out_parent / f"{in_path.stem}{out_suffix}"


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

    small = cv2.resize(
        roi,
        (max(1, w // bw), max(1, h // bh)),
        interpolation=cv2.INTER_LINEAR,
    )
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img_bgr[y1:y2, x1:x2] = pix


def split_mosaic_2x2(frame):
    """
    Split a 2x2 mosaic frame into four subimages:
    [top-left, top-right, bottom-left, bottom-right]
    """
    h, w = frame.shape[:2]
    half_h = h // 2
    half_w = w // 2

    top_left = frame[0:half_h, 0:half_w].copy()
    top_right = frame[0:half_h, half_w:w].copy()
    bottom_left = frame[half_h:h, 0:half_w].copy()
    bottom_right = frame[half_h:h, half_w:w].copy()

    return [top_left, top_right, bottom_left, bottom_right]


def merge_mosaic_2x2(parts):
    """
    Rebuild a 2x2 mosaic from:
    [top-left, top-right, bottom-left, bottom-right]
    """
    top = cv2.hconcat([parts[0], parts[1]])
    bottom = cv2.hconcat([parts[2], parts[3]])
    return cv2.vconcat([top, bottom])


def anonymize_person_heads_in_subimage(
    img,
    result,
    mode: str,
    blur_ksize: int,
    pix_blocks: int,
    head_top_frac: float,
    head_height_frac: float,
    anon_whole_if_none: bool,
):
    h, w = img.shape[:2]

    boxes = []

    # Use the local result passed from process_mosaic_frame(...)
    r0 = result

    if r0.boxes is not None and len(r0.boxes) > 0:
        # Keep detection index so we can fetch the corresponding segmentation mask
        for det_idx, b in enumerate(r0.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = [int(v) for v in b]
            x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, w, h)
            if x2 > x1 and y2 > y1:
                boxes.append((det_idx, x1, y1, x2, y2))

    if not boxes:
        if anon_whole_if_none:
            if mode == "blur":
                img[:] = cv2.GaussianBlur(img, (blur_ksize | 1, blur_ksize | 1), 0)
            else:
                pixelate_roi(img, 0, 0, w, h, blocks=pix_blocks)
        return 0, True

    # Blur/pixelate head region of each person bbox,
    # but limited by the segmentation mask of that person
    for (det_idx, x1, y1, x2, y2) in boxes:
        ph = y2 - y1

        hy1 = y1 + int(ph * head_top_frac)
        hy2 = y1 + int(ph * (head_top_frac + head_height_frac))
        hx1, hx2 = x1, x2

        hx1, hy1, hx2, hy2 = clamp_rect(hx1, hy1, hx2, hy2, w, h)

        person_mask = build_person_mask_from_result(r0, det_idx, h, w)

        apply_effect_limited_by_mask(
            img_bgr=img,
            x1=hx1,
            y1=hy1,
            x2=hx2,
            y2=hy2,
            limiter_mask=person_mask,
            mode=mode,
            blur_ksize=blur_ksize,
            pix_blocks=pix_blocks,
        )

    return len(boxes), False


def process_mosaic_frame(
    frame,
    model: YOLO,
    imgsz: int,
    conf: float,
    iou: float,
    mode: str,
    blur_ksize: int,
    pix_blocks: int,
    head_top_frac: float,
    head_height_frac: float,
    anon_whole_if_none: bool,
    head_pipe
):
    subimgs = split_mosaic_2x2(frame)

    # Run detection in batch for the four views of the mosaic frame
    results = model.predict(
        source=subimgs,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
        classes=[0],  # COCO class 0 = person
    )

    people_in_frame = 0
    no_person_views = 0

    for idx, subimg in enumerate(subimgs):
        n_people, no_person = anonymize_person_heads_in_subimage(
            img=subimg,
            result=results[idx],
            mode=mode,
            blur_ksize=blur_ksize,
            pix_blocks=pix_blocks,
            head_top_frac=head_top_frac,
            head_height_frac=head_height_frac,
            anon_whole_if_none=anon_whole_if_none,
        )
        people_in_frame += n_people
        if no_person:
            no_person_views += 1

    merged = merge_mosaic_2x2(subimgs)
    return merged, people_in_frame, no_person_views


def process_video(
    video_path: Path,
    out_path: Path,
    model: YOLO,
    imgsz: int,
    conf: float,
    iou: float,
    mode: str,
    blur_ksize: int,
    pix_blocks: int,
    head_top_frac: float,
    head_height_frac: float,
    anon_whole_if_none: bool,
    show_stream: bool = True,
    window_name: str = "Mosaic anonymization stream",
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return {
            "frames": 0,
            "people_total": 0,
            "no_person_views": 0,
            "failed": 1,
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    n_frames = 0
    n_people_total = 0
    n_no_person_views = 0
    n_failed = 0

    pbar = tqdm(total=total_frames if total_frames > 0 else None,
                desc=f"Processing {video_path.name}",
                unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            processed_frame, people_count, no_person_views = process_mosaic_frame(
                frame=frame,
                model=model,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                mode=mode,
                blur_ksize=blur_ksize,
                pix_blocks=pix_blocks,
                head_top_frac=head_top_frac,
                head_height_frac=head_height_frac,
                anon_whole_if_none=anon_whole_if_none,
                head_pipe=head_pipe
            )

            writer.write(processed_frame)

            if show_stream:
                cv2.imshow(window_name, processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            n_frames += 1
            n_people_total += people_count
            n_no_person_views += no_person_views

        except Exception as e:
            n_failed += 1
            print(f"Frame processing error in {video_path.name}: {e}")

        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    return {
        "frames": n_frames,
        "people_total": n_people_total,
        "no_person_views": n_no_person_views,
        "failed": n_failed,
    }


def main():
    # -----------------------------
    # CONFIG
    # -----------------------------
    IN_ROOT = Path(r"../videos/")
    OUT_ROOT = Path(r"../videos/blurred/")

    YOLO_WEIGHTS = "models/yolov8n-seg.pt"
    IMGSZ = 960
    CONF = 0.15
    IOU = 0.5

    HEAD_TOP_FRAC = 0.0
    HEAD_HEIGHT_FRAC = 0.35

    MODE = "pixelate"  # "blur" | "pixelate"
    BLUR_KSIZE = 61
    PIX_BLOCKS = 18

    ANON_WHOLE_IF_NONE = False

    SHOW_STREAM = False
    # -----------------------------

    IN_ROOT = IN_ROOT.resolve()
    OUT_ROOT = OUT_ROOT.resolve()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_pipe = head_seg.HumanHeadSegmentationPipeline(device=device)
    model = YOLO(YOLO_WEIGHTS)

    videos = find_videos(IN_ROOT)
    if not videos:
        print(f"No videos found under: {IN_ROOT}")
        return

    global_frames = 0
    global_people = 0
    global_no_person_views = 0
    global_failed = 0

    for video_path in videos:
        out_path = ensure_mirrored_out_path(video_path, IN_ROOT, OUT_ROOT, out_suffix=".mp4")

        stats = process_video(
            video_path=video_path,
            out_path=out_path,
            model=model,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            mode=MODE,
            blur_ksize=BLUR_KSIZE,
            pix_blocks=PIX_BLOCKS,
            head_top_frac=HEAD_TOP_FRAC,
            head_height_frac=HEAD_HEIGHT_FRAC,
            anon_whole_if_none=ANON_WHOLE_IF_NONE,
            show_stream=SHOW_STREAM,
        )

        global_frames += stats["frames"]
        global_people += stats["people_total"]
        global_no_person_views += stats["no_person_views"]
        global_failed += stats["failed"]

        print(f"\nFinished: {video_path.name}")
        print(f"Output: {out_path}")
        print(f"Frames processed: {stats['frames']}")
        print(f"People detected: {stats['people_total']}")
        print(f"Views with no person: {stats['no_person_views']}")
        print(f"Failed frames: {stats['failed']}")

    cv2.destroyAllWindows()

    print("\nDone.")
    print(f"Input root : {IN_ROOT}")
    print(f"Output root: {OUT_ROOT}")
    print(f"Total frames processed: {global_frames}")
    print(f"People detected (total): {global_people}")
    print(f"Views w/ no person: {global_no_person_views}")
    print(f"Failed frames: {global_failed}")


if __name__ == "__main__":
    main()