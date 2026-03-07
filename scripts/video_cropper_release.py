# extract_crops_from_mkv.py
from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
from utils.files import find_files_recursively
from common.consts import CropBoxes, FrameSkipRelease, JpegQualityRelease
from utils.videos import extract_frames_from_video

def main():
    ROOT_DIR = Path(r"../videos/")
    OUTPUT_ROOT = Path(r"../annotations/")

    ROOT_DIR = ROOT_DIR.resolve()
    OUTPUT_ROOT = OUTPUT_ROOT.resolve()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    mkv_files = find_files_recursively(ROOT_DIR, "mkv")
    if not mkv_files:
        print(f"No .mkv files found under: {ROOT_DIR}")
        return

    for video_path in tqdm(mkv_files, desc="Processing videos", unit="video"):
        try:
            extract_frames_from_video(
                video_path=video_path,
                root_dir=ROOT_DIR,
                output_root=OUTPUT_ROOT,
                frame_skip=FrameSkipRelease.VALUE,
                crop_boxes=CropBoxes.VALUE,
                jpeg_quality=JpegQualityRelease.VALUE,
            )
        except Exception as e:
            print(f"\n[ERROR] {video_path} -> {e}")

    print(f"\nDone. Output at: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()