from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

#HEAD MODEL FROM https://github.com/mohammad-hmz/head_detection_cpu/blob/main/head_detection_cpu.py


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


# ------------------------------------------------------------
# Filesystem helpers
# ------------------------------------------------------------
def is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def find_videos(root_dir: Path, exclude_root: Optional[Path] = None) -> List[Path]:
    videos: List[Path] = []
    for path in root_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        if exclude_root is not None and is_under_root(path, exclude_root):
            continue
        videos.append(path)
    return sorted(videos)


def ensure_mirrored_out_path(
    in_path: Path,
    in_root: Path,
    out_root: Path,
    out_suffix: str = ".mp4",
) -> Path:
    rel_parent = in_path.parent.relative_to(in_root)
    out_parent = out_root / rel_parent
    out_parent.mkdir(parents=True, exist_ok=True)
    return out_parent / f"{in_path.stem}{out_suffix}"


# ------------------------------------------------------------
# Image helpers
# ------------------------------------------------------------
def clamp_rect(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


def ensure_odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    if value % 2 == 0:
        value += 1
    return value


def split_mosaic_2x2(frame: np.ndarray) -> List[np.ndarray]:
    h, w = frame.shape[:2]
    half_h = h // 2
    half_w = w // 2
    return [
        frame[0:half_h, 0:half_w].copy(),
        frame[0:half_h, half_w:w].copy(),
        frame[half_h:h, 0:half_w].copy(),
        frame[half_h:h, half_w:w].copy(),
    ]


def merge_mosaic_2x2(parts: List[np.ndarray]) -> np.ndarray:
    top = cv2.hconcat([parts[0], parts[1]])
    bottom = cv2.hconcat([parts[2], parts[3]])
    return cv2.vconcat([top, bottom])


def make_blurred_copy(img_bgr: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.GaussianBlur(img_bgr, (ensure_odd(ksize), ensure_odd(ksize)), 0)


def make_pixelated_copy(img_bgr: np.ndarray, blocks: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return img_bgr.copy()

    blocks = max(2, int(blocks))
    down_w = max(1, w // blocks)
    down_h = max(1, h // blocks)

    reduced = cv2.resize(img_bgr, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(reduced, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_effect_on_mask(
    img_bgr: np.ndarray,
    binary_mask: Optional[np.ndarray],
    mode: str,
    blur_ksize: int,
    pixel_blocks: int,
) -> int:
    if binary_mask is None or not np.any(binary_mask):
        return 0

    ys, xs = np.where(binary_mask > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    roi = img_bgr[y1:y2, x1:x2]
    roi_mask = binary_mask[y1:y2, x1:x2] > 0

    if roi.size == 0 or not np.any(roi_mask):
        return 0

    if mode == "blur":
        processed = make_blurred_copy(roi, blur_ksize)
    else:
        processed = make_pixelated_copy(roi, pixel_blocks)

    roi[roi_mask] = processed[roi_mask]
    return int(np.count_nonzero(roi_mask))


# ------------------------------------------------------------
# YOLO result helpers
# ------------------------------------------------------------
def extract_head_boxes(result, h: int, w: int) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []

    if result.boxes is None or len(result.boxes) == 0:
        return boxes

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    for box in xyxy:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, w, h)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return boxes


def build_union_person_mask(result, h: int, w: int, person_class_id: int = 0) -> Optional[np.ndarray]:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    if result.masks is None or len(result.masks.xy) == 0:
        return None

    union_mask = np.zeros((h, w), dtype=np.uint8)
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    for det_idx, cls_id in enumerate(classes):
        if cls_id != person_class_id:
            continue
        if det_idx >= len(result.masks.xy):
            continue

        polygon = result.masks.xy[det_idx]
        if polygon is None or len(polygon) == 0:
            continue

        pts = np.round(polygon).astype(np.int32)
        cv2.fillPoly(union_mask, [pts], 255)

    if not np.any(union_mask):
        return None

    return union_mask


def build_intersection_mask(
    head_boxes: List[Tuple[int, int, int, int]],
    person_mask: Optional[np.ndarray],
    h: int,
    w: int,
) -> Optional[np.ndarray]:
    if person_mask is None or not np.any(person_mask) or not head_boxes:
        return None

    out_mask = np.zeros((h, w), dtype=np.uint8)

    for x1, y1, x2, y2 in head_boxes:
        head_box_mask = np.zeros((h, w), dtype=np.uint8)
        head_box_mask[y1:y2, x1:x2] = 255
        out_mask = cv2.bitwise_or(out_mask, cv2.bitwise_and(person_mask, head_box_mask))

    if not np.any(out_mask):
        return None

    return out_mask


# ------------------------------------------------------------
# Core processing
# ------------------------------------------------------------
def anonymize_subimage(
    img: np.ndarray,
    head_result,
    seg_result,
    mode: str,
    blur_ksize: int,
    pixel_blocks: int,
    fallback_upper_portion: float = 0.10,
) -> Tuple[int, int, int]:
    h, w = img.shape[:2]

    head_boxes = extract_head_boxes(head_result, h, w)
    person_mask = build_union_person_mask(seg_result, h, w, person_class_id=0)
    intersection_mask = build_intersection_mask(head_boxes, person_mask, h, w)

    mask_to_apply = None

    # prioridade 1: interseção entre cabeça e segmentação de pessoa
    if intersection_mask is not None and np.any(intersection_mask):
        mask_to_apply = intersection_mask

    # prioridade 2: topo da máscara da pessoa
    elif person_mask is not None and np.any(person_mask):
        mask_to_apply = build_upper_person_fallback_mask(
            person_mask=person_mask,
            upper_portion=fallback_upper_portion,
        )

    # prioridade 3: se a segmentação falhar, usa a própria máscara das head boxes
    elif len(head_boxes) > 0:
        mask_to_apply = build_head_boxes_mask(
            head_boxes=head_boxes,
            h=h,
            w=w,
        )

    affected_pixels = apply_effect_on_mask(
        img_bgr=img,
        binary_mask=mask_to_apply,
        mode=mode,
        blur_ksize=blur_ksize,
        pixel_blocks=pixel_blocks,
    )

    people_detected = 0 if person_mask is None else 1
    return len(head_boxes), people_detected, affected_pixels



def process_mosaic_frame(
    frame: np.ndarray,
    head_model: YOLO,
    seg_model: YOLO,
    head_imgsz: int,
    seg_imgsz: int,
    head_conf: float,
    seg_conf: float,
    head_iou: float,
    seg_iou: float,
    mode: str,
    blur_ksize: int,
    pixel_blocks: int,
    yolo_device,
    use_half: bool,
    fallback_upper_portion: float
) -> Tuple[np.ndarray, int, int, int]:
    subimgs = split_mosaic_2x2(frame)

    head_results = head_model.predict(
        source=subimgs,
        imgsz=head_imgsz,
        conf=head_conf,
        iou=head_iou,
        verbose=False,
        device=yolo_device,
        half=use_half,
    )

    seg_results = seg_model.predict(
        source=subimgs,
        imgsz=seg_imgsz,
        conf=seg_conf,
        iou=seg_iou,
        verbose=False,
        classes=[0],
        device=yolo_device,
        half=use_half,
        retina_masks=False,
    )

    total_heads = 0
    total_people_views = 0
    total_affected_pixels = 0

    for idx, subimg in enumerate(subimgs):
        heads_detected, people_detected, affected_pixels = anonymize_subimage(
            img=subimg,
            head_result=head_results[idx],
            seg_result=seg_results[idx],
            mode=mode,
            blur_ksize=blur_ksize,
            pixel_blocks=pixel_blocks,
            fallback_upper_portion=0.10,
        )
        total_heads += heads_detected
        total_people_views += people_detected
        total_affected_pixels += affected_pixels

    return merge_mosaic_2x2(subimgs), total_heads, total_people_views, total_affected_pixels

def build_upper_person_fallback_mask(
    person_mask: Optional[np.ndarray],
    upper_portion: float = 0.10,
) -> Optional[np.ndarray]:
    if person_mask is None or not np.any(person_mask):
        return None

    ys, xs = np.where(person_mask > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    person_h = y2 - y1
    if person_h <= 0:
        return None

    cutoff_y = y1 + max(1, int(person_h * upper_portion))

    fallback_mask = np.zeros_like(person_mask, dtype=np.uint8)
    fallback_mask[y1:cutoff_y, x1:x2] = person_mask[y1:cutoff_y, x1:x2]

    if not np.any(fallback_mask):
        return None

    return fallback_mask

from pathlib import Path
from typing import List
import cv2
from tqdm import tqdm
from ultralytics import YOLO

import cv2
from pathlib import Path
from typing import List
from data.crop2pts import Crop2Pts
from utils.files import ensure_cam_dirs
from utils.maths import clamp_crop, is_valid_crop

# Assume que já existem no seu projeto:
# - Crop2Pts
# - clamp_crop
# - is_valid_crop
# - ensure_cam_dirs
# - process_mosaic_frame

from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO


def _fourcc_int_to_str(fourcc_value: int) -> str:
    try:
        return "".join([
            chr((fourcc_value >> 0) & 0xFF),
            chr((fourcc_value >> 8) & 0xFF),
            chr((fourcc_value >> 16) & 0xFF),
            chr((fourcc_value >> 24) & 0xFF),
        ])
    except Exception:
        return ""


def _is_reasonable_fourcc(codec: str) -> bool:
    if not codec or len(codec) != 4:
        return False
    # evita lixo / nulos
    return all(32 <= ord(c) <= 126 for c in codec)


def _open_video_writer(out_path: Path, fps: float, width: int, height: int, input_codec: str = ""):
    suffix = out_path.suffix.lower()

    candidates = []

    # tenta primeiro o codec do vídeo de entrada, se parecer válido
    if _is_reasonable_fourcc(input_codec):
        candidates.append(input_codec)

    # depois tenta codecs comuns por container
    if suffix == ".mkv":
        candidates.extend(["FFV1", "XVID", "MJPG", "mp4v"])
    elif suffix == ".mp4":
        candidates.extend(["mp4v", "avc1", "H264", "XVID", "MJPG"])
    elif suffix == ".avi":
        candidates.extend(["XVID", "MJPG", "FFV1", "mp4v"])
    else:
        candidates.extend(["FFV1", "XVID", "MJPG", "mp4v"])

    tried = []
    for codec in candidates:
        if codec in tried:
            continue
        tried.append(codec)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, codec

        writer.release()

    raise RuntimeError(
        f"Could not open VideoWriter for {out_path}. "
        f"Tried codecs: {tried}"
    )


def process_video(
    video_path: Path,
    out_path: Path,
    head_model: YOLO,
    seg_model: YOLO,
    head_imgsz: int,
    seg_imgsz: int,
    head_conf: float,
    seg_conf: float,
    head_iou: float,
    seg_iou: float,
    mode: str,
    blur_ksize: int,
    pixel_blocks: int,
    yolo_device,
    use_half: bool,
    show_stream: bool,
    write_output: bool,
    window_name: str = "Human face anonymization",
    fallback_upper_portion: float = 0.1
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return {"frames": 0, "heads": 0, "people_views": 0, "affected_pixels": 0, "failed": 1}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    input_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC)) or 0
    input_codec = _fourcc_int_to_str(input_fourcc_int)

    writer = None
    selected_codec = None

    if write_output:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer, selected_codec = _open_video_writer(
            out_path=out_path,
            fps=fps,
            width=width,
            height=height,
            input_codec=input_codec
        )
        print(
            f"Writing output: {out_path} | "
            f"fps={fps:.3f} | size=({width}x{height}) | "
            f"input_codec={repr(input_codec)} | output_codec={selected_codec}"
        )

    frames_processed = 0
    heads_detected_total = 0
    people_views_total = 0
    affected_pixels_total = 0
    failed_frames = 0

    pbar = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc=f"Processing {video_path.name}",
        unit="frame"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        try:
            processed_frame, heads_count, people_views, affected_pixels = process_mosaic_frame(
                frame=frame,
                head_model=head_model,
                seg_model=seg_model,
                head_imgsz=head_imgsz,
                seg_imgsz=seg_imgsz,
                head_conf=head_conf,
                seg_conf=seg_conf,
                head_iou=head_iou,
                seg_iou=seg_iou,
                mode=mode,
                blur_ksize=blur_ksize,
                pixel_blocks=pixel_blocks,
                yolo_device=yolo_device,
                use_half=use_half,
                fallback_upper_portion=fallback_upper_portion
            )

            # garante compatibilidade com writer
            if processed_frame.ndim == 2:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            elif processed_frame.ndim == 3 and processed_frame.shape[2] == 4:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2BGR)

            if processed_frame.shape[1] != width or processed_frame.shape[0] != height:
                processed_frame = cv2.resize(processed_frame, (width, height))

            if writer is not None:
                writer.write(processed_frame)

            if show_stream:
                try:
                    cv2.imshow(window_name, cv2.resize(processed_frame, (1024, 720)))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                except cv2.error as exc:
                    print(f"Display unavailable, disabling stream: {exc}")
                    show_stream = False

            frames_processed += 1
            heads_detected_total += heads_count
            people_views_total += people_views
            affected_pixels_total += affected_pixels

        except Exception as exc:
            failed_frames += 1
            print(f"Frame processing error in {video_path.name}: {exc}")

            # Para não quebrar a engenharia reversa / reconstrução,
            # escreve o frame original quando o processamento falhar.
            if writer is not None:
                writer.write(frame)

            frames_processed += 1

        pbar.update(1)

    pbar.close()
    cap.release()

    if writer is not None:
        writer.release()

    if show_stream:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return {
        "frames": frames_processed,
        "heads": heads_detected_total,
        "people_views": people_views_total,
        "affected_pixels": affected_pixels_total,
        "failed": failed_frames,
    }


def build_upper_person_fallback_mask(
    person_mask: Optional[np.ndarray],
    upper_portion: float = 0.10,
) -> Optional[np.ndarray]:
    if person_mask is None or not np.any(person_mask):
        return None

    ys, xs = np.where(person_mask > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    person_h = y2 - y1
    if person_h <= 0:
        return None

    cutoff_y = y1 + max(1, int(person_h * upper_portion))

    fallback_mask = np.zeros_like(person_mask, dtype=np.uint8)
    fallback_mask[y1:cutoff_y, x1:x2] = person_mask[y1:cutoff_y, x1:x2]

    if not np.any(fallback_mask):
        return None

    return fallback_mask


def build_head_boxes_mask(
    head_boxes: List[Tuple[int, int, int, int]],
    h: int,
    w: int,
) -> Optional[np.ndarray]:
    if not head_boxes:
        return None

    head_mask = np.zeros((h, w), dtype=np.uint8)

    for x1, y1, x2, y2 in head_boxes:
        head_mask[y1:y2, x1:x2] = 255

    if not np.any(head_mask):
        return None

    return head_mask


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def main() -> None:
    # Input / output
    in_root = Path(r"../videos/").resolve()
    out_root = Path(r"../videos/blurred/").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Models
    head_weights = "models/head_nano.pt"
    seg_weights = "models/yolov8n-seg.pt"

    # Inference setup
    head_imgsz = 960
    seg_imgsz = 960
    head_conf = 0.05
    seg_conf = 0.05
    head_iou = 0.50
    seg_iou = 0.50
    fallback_upper_portion = 0.10

    # Effect setup
    mode = "pixelate"   # "pixelate" or "blur"
    blur_ksize = 61
    pixel_blocks = 18

    # Runtime
    show_stream = True
    write_output = True

    use_cuda = torch.cuda.is_available()
    yolo_device = 0 if use_cuda else "cpu"
    use_half = use_cuda

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    head_model = YOLO(head_weights)
    seg_model = YOLO(seg_weights)

    videos = find_videos(in_root, exclude_root=out_root)
    if not videos:
        print(f"No videos found under: {in_root}")
        return

    global_frames = 0
    global_heads = 0
    global_people_views = 0
    global_affected_pixels = 0
    global_failed = 0

    print(f"Device: {'CUDA' if use_cuda else 'CPU'}")
    print(f"Found videos: {len(videos)}")

    for video_path in videos:
        out_path = ensure_mirrored_out_path(video_path, in_root, out_root, out_suffix=".mp4")

        stats = process_video(
            video_path=video_path,
            out_path=out_path,
            head_model=head_model,
            seg_model=seg_model,
            head_imgsz=head_imgsz,
            seg_imgsz=seg_imgsz,
            head_conf=head_conf,
            seg_conf=seg_conf,
            head_iou=head_iou,
            seg_iou=seg_iou,
            mode=mode,
            blur_ksize=blur_ksize,
            pixel_blocks=pixel_blocks,
            yolo_device=yolo_device,
            use_half=use_half,
            show_stream=show_stream,
            write_output=write_output,
            fallback_upper_portion=fallback_upper_portion
        )

        global_frames += stats["frames"]
        global_heads += stats["heads"]
        global_people_views += stats["people_views"]
        global_affected_pixels += stats["affected_pixels"]
        global_failed += stats["failed"]

        print(f"\nFinished: {video_path.name}")
        print(f"Output: {out_path}")
        print(f"Frames processed: {stats['frames']}")
        print(f"Head boxes detected: {stats['heads']}")
        print(f"Views with person mask: {stats['people_views']}")
        print(f"Affected pixels: {stats['affected_pixels']}")
        print(f"Failed frames: {stats['failed']}")

    cv2.destroyAllWindows()

    print("\nDone.")
    print(f"Input root: {in_root}")
    print(f"Output root: {out_root}")
    print(f"Total frames processed: {global_frames}")
    print(f"Total head boxes detected: {global_heads}")
    print(f"Total views with person mask: {global_people_views}")
    print(f"Total affected pixels: {global_affected_pixels}")
    print(f"Total failed frames: {global_failed}")


if __name__ == "__main__":
    main()
