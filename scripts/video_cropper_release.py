from __future__ import annotations
from tqdm import tqdm
from utils.files import find_files_recursively
from common.consts import CropBoxes, FrameSkipRelease, JpegQualityRelease, VideoDirPath, AnnotationsDirPath
from utils.videos import extract_frames_from_video

def main():
    root_dir = VideoDirPath.VALUE
    output_dir = AnnotationsDirPath.VALUE
    output_dir.mkdir(parents=True, exist_ok=True)

    mkv_files = find_files_recursively(root_dir, "mp4")
    if not mkv_files:
        print(f"No .mkv files found under: {root_dir}")
        return

    for video_path in tqdm(mkv_files, desc="Processing videos", unit="video"):
        try:
            extract_frames_from_video(
                video_path=video_path,
                root_dir=root_dir,
                output_root=output_dir,
                frame_skip=FrameSkipRelease.VALUE,
                crop_boxes=CropBoxes.VALUE,
                jpeg_quality=JpegQualityRelease.VALUE,
            )
        except Exception as e:
            print(f"\n[ERROR] {video_path} -> {e}")

    print(f"\nDone. Output at: {output_dir}")


if __name__ == "__main__":
    main()