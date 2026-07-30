from __future__ import annotations

import numpy as np

from .surface import SRCALPHA, Surface, _rgba_array


def array3d(surface):
    return np.transpose(surface.pixels[..., :3], (1, 0, 2)).copy()


def pixels3d(surface):
    return np.transpose(surface.pixels[..., :3], (1, 0, 2))


def array_alpha(surface):
    return surface.pixels[..., 3].T.copy()


def pixels_alpha(surface):
    if not surface.get_flags() & SRCALPHA:
        raise ValueError("unsupported colormasks for alpha reference array")
    return surface.pixels[..., 3].T


def make_surface(array):
    pixels = np.asarray(array)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("must be a valid 3d array")
    return Surface._from_pixels(np.transpose(pixels, (1, 0, 2)), 0)


def blit_array(surface, array):
    pixels = _rgba_array(array)
    expected = (surface.get_width(), surface.get_height())
    if pixels.shape[:2] != expected:
        raise ValueError("array must match surface dimensions")
    if pixels.ndim == 3 and pixels.shape[2] in (3, 4):
        surface.pixels[..., : pixels.shape[2]] = np.transpose(pixels, (1, 0, 2))
    else:
        raise ValueError("expected an RGB or RGBA array")
