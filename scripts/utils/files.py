import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "common")))

from pathlib import Path
from typing import List
import pandas as pd
from consts import SARDirDataTemplate, SARCDirDataTemplate, MultiPersonVideoDirPath, LogMissingVideoFilePath, GenericMultiPersonDirPath, MultiPersonAnnotationsDirPath, LogFramesQuantityAssertionFilePath

def find_files_recursively(root_dir: Path, file_type: str) -> List[Path]:
    return sorted(root_dir.rglob(f"*.{file_type}"))

def ensure_cam_dirs(parent_dir: Path, video_stem: str) -> List[Path]:
    out_dirs: List[Path] = []
    for cam_idx in range(1, 5):
        d = parent_dir / f"{video_stem}_cam{cam_idx}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs.append(d)
    return out_dirs

def get_all_video_path_from_data_info(df_data_info : pd.DataFrame, dump_report : bool = True) -> List[Path]:
    all_paths = []
    videos_quantity = df_data_info.shape[0]    
    for data_idx in range(videos_quantity):
        data = df_data_info.loc[data_idx].to_dict()
        video_fname_relative = SARDirDataTemplate.VALUE.format(data["SUBJECT"], data["ACTIVITY"], data["ROUTINE"])
        video_path_relative = Path(video_fname_relative)
        video_path_absolute = (MultiPersonVideoDirPath.VALUE / video_path_relative).resolve()
        if video_path_absolute not in all_paths:
            all_paths.append(video_path_absolute)
    if dump_report:
        try_write_list_in_file(all_paths, LogMissingVideoFilePath.VALUE)
    return all_paths

def try_write_list_in_file(list : List, file_path : Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    print(file_path)
    try:
        with file_path.open("w", encoding="utf-8") as f:
            for element in list:
                f.write(str(element) + "\n")
    except:
        print("Elements in list does not support cast to string!")

def get_file_type_quantity_from_folder(folder_path : Path, file_type : str) -> int:
    files = list(folder_path.glob(f"*.{file_type}"))
    return len(files)

def check_data_integrity_extracted_frames(df_data_info : pd.DataFrame, subset_path : Path = GenericMultiPersonDirPath.VALUE, dump_report : bool = True):
    from utils.videos import get_video_frames_quantity

    all_report = []
    if subset_path == GenericMultiPersonDirPath.VALUE:
        subset_absolute_path = MultiPersonAnnotationsDirPath.VALUE

    check_quantity = df_data_info.shape[0]    
    for data_idx in range(check_quantity):
        data = df_data_info.loc[data_idx].to_dict()
        video_fname_relative = SARDirDataTemplate.VALUE.format(data["SUBJECT"], data["ACTIVITY"], data["ROUTINE"])
        video_path_absolute = (MultiPersonVideoDirPath.VALUE / Path(video_fname_relative)).resolve()
        expected_frame_quantity = get_video_frames_quantity(video_path_absolute)
        frames_folder_fname_relative = SARCDirDataTemplate.VALUE.format(data["SUBJECT"], data["ACTIVITY"], data["ROUTINE"], data["CAMERA"])
        frames_folder_path_absolute = (subset_absolute_path / Path(frames_folder_fname_relative)).resolve()
        current_frames_quantity = get_file_type_quantity_from_folder(frames_folder_path_absolute, "jpg")
        assert_result = "SUCESS" if expected_frame_quantity == current_frames_quantity else "FAILED"
        all_report.append(f"Video {video_fname_relative} has {expected_frame_quantity} frames - Frame folder {frames_folder_fname_relative} has {current_frames_quantity} - Assertion {assert_result}!")
    
    print(all_report)
    if dump_report:
        try_write_list_in_file(all_report, LogFramesQuantityAssertionFilePath.VALUE)

def delete_annotations_missing_information(df_data_info : pd.DataFrame, subset_path : Path = GenericMultiPersonDirPath.VALUE):
    if subset_path == GenericMultiPersonDirPath.VALUE:
        subset_absolute_path = MultiPersonAnnotationsDirPath.VALUE

    check_quantity = df_data_info.shape[0]    
    for data_idx in range(check_quantity):
        data = df_data_info.loc[data_idx].to_dict()
        frames_folder_fname_relative = SARCDirDataTemplate.VALUE.format(data["SUBJECT"], data["ACTIVITY"], data["ROUTINE"], data["CAMERA"])
        frames_folder_path_absolute = (subset_absolute_path / Path(frames_folder_fname_relative)).resolve()
        annotations = find_files_recursively(frames_folder_path_absolute, file_type="json")

        for annotation in annotations:
            annotation.unlink()
    
    print("Done!")
    