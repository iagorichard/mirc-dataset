import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))

from typing import Any, List
from pathlib import Path
from crop2pts import Crop2Pts

class ConstantBase:
    VALUE: Any

class JpegQualityRelease(ConstantBase): 
    VALUE : int = 95

class JpegQualityTests(ConstantBase): 
    VALUE : int = 100

class FrameSkipRelease(ConstantBase): 
    VALUE : int  = 15

class FrameSkipTests(ConstantBase):
    VALUE : int = 1

class CropBoxes(ConstantBase):
    VALUE : List[Crop2Pts] = [
                ((0,    0),    (720, 1280)),   # Camera 1
                ((0,    1320), (720, 2600)),   # Camera 2
                ((780,  0),    (1500, 1280)),  # Camera 3
                ((780,  1320), (1500, 2600)),  # Camera 4
            ]
    
class VideoDirPath(ConstantBase): 
    VALUE : Path = Path(r"../videos/").resolve()

class AnnotationsDirPath(ConstantBase): 
    VALUE : Path = Path(r"../annotations/").resolve()

class MultiPersonVideoDirPath(ConstantBase): 
    VALUE : Path = (VideoDirPath.VALUE / Path(r"multi-person/")).resolve()

class MultiPersonVideoDirPath(ConstantBase): 
    VALUE : Path = (VideoDirPath.VALUE / Path(r"multi-person/")).resolve()

class ScriptLogsDirPath(ConstantBase): 
    VALUE : Path = Path("outputs/").resolve()

class LogMissingVideoFilePath(ConstantBase): 
    VALUE : Path = (ScriptLogsDirPath.VALUE / Path("videos_with_missing_annotations.dtxt")).resolve()

class SARDirDataTemplate(ConstantBase):
    VALUE : str = "subject{0}/activity{1}/routine{2:02d}.mkv"