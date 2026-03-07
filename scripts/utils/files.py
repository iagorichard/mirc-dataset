import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "common")))

from pathlib import Path
from typing import List
import pandas as pd
from consts import SARDirDataTemplate, MultiPersonVideoDirPath, LogMissingVideoFilePath

def find_files_recursively(root_dir: Path, file_type: str) -> List[Path]:
    return sorted(root_dir.rglob(f"*.{file_type}"))

def ensure_cam_dirs(parent_dir: Path, video_stem: str) -> List[Path]:
    out_dirs: List[Path] = []
    for cam_idx in range(1, 5):
        d = parent_dir / f"{video_stem}_cam{cam_idx}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs.append(d)
    return out_dirs

def get_all_video_path_from_data_info(df_data_info : pd.DataFrame, dump_report : bool = True) -> List[str]:
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