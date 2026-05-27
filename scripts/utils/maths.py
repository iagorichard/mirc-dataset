import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))

from point import Point
from crop2pts import Crop2Pts

def clamp_point(p: Point, w: int, h: int) -> Point:
    y, x = p
    y = max(0, min(y, h))
    x = max(0, min(x, w))
    return y, x

def clamp_crop(crop: Crop2Pts, w: int, h: int) -> Crop2Pts:
    (y1, x1), (y2, x2) = crop
    (y1, x1) = clamp_point((y1, x1), w, h)
    (y2, x2) = clamp_point((y2, x2), w, h)
    return (y1, x1), (y2, x2)

def is_valid_crop(crop: Crop2Pts) -> bool:
    (y1, x1), (y2, x2) = crop
    return (y2 > y1) and (x2 > x1)