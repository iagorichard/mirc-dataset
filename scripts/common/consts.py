import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))

from typing import Any, List
from crop2pts import Crop2Pts

class ConstantBase:
    VALUE: Any

class JpegQualityRelease(ConstantBase): 
    VALUE = 95

class JpegQualityReleaseTests(ConstantBase): 
    VALUE = 100

class FrameSkipRelease(ConstantBase): 
    VALUE = 15

class FrameSkipTests(ConstantBase):
    VALUE = 1

class CropBoxes:
    VALUE : List[Crop2Pts] = [
                ((0,    0),    (720, 1280)),   # Camera 1
                ((0,    1320), (720, 2600)),   # Camera 2
                ((780,  0),    (1500, 1280)),  # Camera 3
                ((780,  1320), (1500, 2600)),  # Camera 4
            ]
            