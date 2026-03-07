from pathlib import Path
from typing import List

def find_files_recursively(root_dir: Path, file_type: str) -> List[Path]:
    return sorted(root_dir.rglob(f"*.{file_type}"))

def ensure_cam_dirs(parent_dir: Path, video_stem: str) -> List[Path]:
    out_dirs: List[Path] = []
    for cam_idx in range(1, 5):
        d = parent_dir / f"{video_stem}_cam{cam_idx}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs.append(d)
    return out_dirs