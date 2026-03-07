from __future__ import annotations
from utils.annotations import get_missing_video_annotation_data_info
from tqdm import tqdm
from common.consts import CropBoxes, FrameSkipTests, JpegQualityTests, VideoDirPath, AnnotationsDirPath
from utils.videos import extract_frames_from_video
from utils.files import get_all_video_path_from_data_info

def main():
    root_dir = VideoDirPath.VALUE
    output_dir = AnnotationsDirPath.VALUE
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_data = get_missing_video_annotation_data_info()
    missing_videos = get_all_video_path_from_data_info(missing_data)

    return

    if not missing_videos:
        print(f"No .mkv files found under: {root_dir}")
        return

    for video_path in tqdm(missing_videos, desc="Processing videos", unit="video"):
        try:
            extract_frames_from_video(
                video_path=video_path,
                root_dir=root_dir,
                output_root=output_dir,
                frame_skip=FrameSkipTests.VALUE,
                crop_boxes=CropBoxes.VALUE,
                jpeg_quality=JpegQualityTests.VALUE,
            )
        except Exception as e:
            print(f"\n[ERROR] {video_path} -> {e}")

    print(f"\nDone. Output at: {output_dir}")


if __name__ == "__main__":
    main()