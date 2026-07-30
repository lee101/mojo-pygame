"""A pygame-shaped API for Mojo-accelerated pixel, collision, and PCM work."""

from .rect import Rect
from .surface import (
    BLEND_ADD,
    BLEND_ALPHA_SDL2,
    BLEND_MAX,
    BLEND_MIN,
    BLEND_MULT,
    BLEND_PREMULTIPLIED,
    BLEND_RGBA_ADD,
    BLEND_RGBA_MAX,
    BLEND_RGBA_MIN,
    BLEND_RGBA_MULT,
    BLEND_RGBA_SUB,
    BLEND_SUB,
    SRCALPHA,
    Surface,
    from_array,
)
from . import mask, mixer, sprite, surfarray

__version__ = "0.1.0"


def init():
    mixer.init()
    return 1, 0


def quit():
    mixer.quit()
