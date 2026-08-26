from __future__ import annotations

import numpy as np

from . import _lib
from ._parallel import ranges as _parallel_ranges
from .rect import Rect

SRCALPHA = 65536
BLEND_ADD = 1
BLEND_SUB = 2
BLEND_MULT = 3
BLEND_MIN = 4
BLEND_MAX = 5
BLEND_RGBA_ADD = 6
BLEND_RGBA_SUB = 7
BLEND_RGBA_MULT = 8
BLEND_RGBA_MIN = 9
BLEND_RGBA_MAX = 16
BLEND_PREMULTIPLIED = 17
BLEND_ALPHA_SDL2 = 18

_BLEND_FLAGS = {
    0,
    BLEND_ADD,
    BLEND_SUB,
    BLEND_MULT,
    BLEND_MIN,
    BLEND_MAX,
    BLEND_RGBA_ADD,
    BLEND_RGBA_SUB,
    BLEND_RGBA_MULT,
    BLEND_RGBA_MIN,
    BLEND_RGBA_MAX,
    BLEND_PREMULTIPLIED,
    BLEND_ALPHA_SDL2,
}

_PARALLEL_BLIT_PIXELS = 1_048_576
_PARALLEL_BLEND_PIXELS = 8_388_608


def _color(value, default_alpha=255):
    if isinstance(value, (int, np.integer)):
        number = int(value)
        return (
            (number >> 16) & 255,
            (number >> 8) & 255,
            number & 255,
            (number >> 24) & 255,
        )
    values = tuple(int(component) for component in value)
    if len(values) == 3:
        values += (default_alpha,)
    if len(values) != 4 or any(component < 0 or component > 255 for component in values):
        raise ValueError("invalid color argument")
    return values


def _rgba_array(pixels):
    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("pixels must have shape (height, width, 3 or 4)")
    if array.dtype.kind not in "bu":
        raise TypeError("pixel arrays must contain unsigned integers")
    if array.dtype != np.uint8 and array.size and int(array.max()) > 255:
        raise ValueError("pixel values must be in the range 0..255")
    return np.ascontiguousarray(array, dtype=np.uint8)


class Surface:
    def __init__(self, size, flags=0, depth=0, masks=None):
        width, height = (int(v) for v in size)
        if width < 0 or height < 0:
            raise ValueError("Invalid resolution for Surface")
        if depth not in (0, 32):
            raise ValueError("only 32-bit surfaces are supported")
        if masks is not None:
            raise ValueError("custom pixel masks are not supported")
        self._pixels = np.zeros((height, width, 4), dtype=np.uint8)
        self._pixels[..., 3] = 0 if flags & SRCALPHA else 255
        self._flags = int(flags) & SRCALPHA
        self._alpha = 255 if flags & SRCALPHA else None
        self._colorkey = None
        self._clip = Rect(0, 0, width, height)

    @classmethod
    def _from_pixels(cls, pixels, flags=SRCALPHA):
        array = _rgba_array(pixels)
        result = cls((array.shape[1], array.shape[0]), flags)
        result._pixels[..., : array.shape[2]] = array
        if array.shape[2] == 3 or not flags & SRCALPHA:
            result._pixels[..., 3] = 255
        return result

    @property
    def pixels(self):
        return self._pixels

    def get_size(self):
        return self._pixels.shape[1], self._pixels.shape[0]

    def get_width(self):
        return self._pixels.shape[1]

    def get_height(self):
        return self._pixels.shape[0]

    def get_rect(self, **kwargs):
        rect = Rect(0, 0, *self.get_size())
        for name, value in kwargs.items():
            if not hasattr(rect, name):
                raise TypeError(f"invalid rect assignment: {name}")
            setattr(rect, name, value)
        return rect

    def get_flags(self):
        return self._flags

    def get_bitsize(self):
        return 32

    def get_bytesize(self):
        return 4

    def get_pitch(self):
        return self.get_width() * 4

    def get_masks(self):
        return 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000 if self._flags & SRCALPHA else 0

    def get_shifts(self):
        return 16, 8, 0, 24 if self._flags & SRCALPHA else 0

    def get_losses(self):
        return 0, 0, 0, 0 if self._flags & SRCALPHA else 8

    def copy(self):
        result = Surface._from_pixels(self._pixels.copy(), self._flags)
        result._alpha = self._alpha
        result._colorkey = self._colorkey
        result._clip = self._clip.copy()
        return result

    def convert(self, surface=None):
        result = self.copy()
        result._flags &= ~SRCALPHA
        result._pixels[..., 3] = 255
        return result

    def convert_alpha(self, surface=None):
        result = self.copy()
        result._flags |= SRCALPHA
        return result

    def set_alpha(self, value, flags=0):
        if value is None:
            self._alpha = None
        else:
            self._alpha = min(255, max(0, int(value)))

    def get_alpha(self):
        return self._alpha

    def set_colorkey(self, color, flags=0):
        self._colorkey = None if color is None else _color(color)[:3]

    def get_colorkey(self):
        return None if self._colorkey is None else (*self._colorkey, 255)

    def set_clip(self, rect=None):
        old = self._clip.copy()
        bounds = Rect(0, 0, *self.get_size())
        self._clip = bounds if rect is None else bounds.clip(rect)
        return old

    def get_clip(self):
        return self._clip.copy()

    def get_at(self, pos):
        x, y = (int(v) for v in pos)
        if not (0 <= x < self.get_width() and 0 <= y < self.get_height()):
            raise IndexError("pixel index out of range")
        return tuple(int(v) for v in self._pixels[y, x])

    def set_at(self, pos, color):
        x, y = (int(v) for v in pos)
        if not (0 <= x < self.get_width() and 0 <= y < self.get_height()):
            raise IndexError("pixel index out of range")
        rgba = _color(color)
        if not self._flags & SRCALPHA:
            rgba = (*rgba[:3], 255)
        self._pixels[y, x] = rgba

    def map_rgb(self, color):
        r, g, b, a = _color(color)
        return (a << 24) | (r << 16) | (g << 8) | b

    def unmap_rgb(self, value):
        return _color(value)

    def fill(self, color, rect=None, special_flags=0):
        requested = self._clip.copy() if rect is None else Rect(rect)
        if (
            requested.w <= 0
            or requested.h <= 0
            or requested.right <= self._clip.left
            or requested.bottom <= self._clip.top
            or requested.left >= self._clip.right
            or requested.top >= self._clip.bottom
        ):
            return Rect(0, 0, 0, 0)
        x = max(requested.x, self._clip.x)
        y = max(requested.y, self._clip.y)
        target = Rect(
            x,
            y,
            min(requested.w, self._clip.right - x),
            min(requested.h, self._clip.bottom - y),
        )
        if target.w <= 0 or target.h <= 0:
            return Rect(0, 0, 0, 0)
        rgba = np.asarray(_color(color), dtype=np.uint8)
        view = self._pixels[target.y : target.bottom, target.x : target.right]
        if special_flags == 0:
            view[...] = rgba
            if not self._flags & SRCALPHA:
                view[..., 3] = 255
        else:
            source = Surface((target.w, target.h), SRCALPHA)
            source._pixels[...] = rgba
            self.blit(source, target, special_flags=special_flags)
        return target

    def blit(self, source, dest, area=None, special_flags=0):
        if not isinstance(source, Surface):
            raise TypeError("source must be a Surface")
        if special_flags not in _BLEND_FLAGS:
            raise ValueError("Unsupported blit special flag")
        dx, dy = (dest.x, dest.y) if hasattr(dest, "x") else (int(dest[0]), int(dest[1]))
        source_rect = Rect(0, 0, *source.get_size()) if area is None else Rect(area)
        source_bounds = Rect(0, 0, *source.get_size())
        clipped_source = source_rect.clip(source_bounds)
        dx += clipped_source.x - source_rect.x
        dy += clipped_source.y - source_rect.y
        destination = Rect(dx, dy, clipped_source.w, clipped_source.h)
        clipped_destination = destination.clip(self._clip)
        sx = clipped_source.x + clipped_destination.x - destination.x
        sy = clipped_source.y + clipped_destination.y - destination.y
        if clipped_destination.w <= 0 or clipped_destination.h <= 0:
            return Rect(clipped_destination.x, clipped_destination.y, 0, 0)

        source_pixels = source._pixels
        source_stride = source_pixels.strides[0]
        if np.shares_memory(source._pixels, self._pixels):
            source_pixels = np.ascontiguousarray(
                source._pixels[
                    sy : sy + clipped_destination.h,
                    sx : sx + clipped_destination.w,
                ].copy()
            )
            source_stride = source_pixels.strides[0]
            sx = sy = 0

        key = source._colorkey or (0, 0, 0)
        def blit_rows(row_start, row_end):
            _lib.lib().mpg_blit_rgba(
                _lib.addr(source_pixels),
                _lib.addr(self._pixels),
                source_stride,
                self._pixels.strides[0],
                sx,
                sy + row_start,
                clipped_destination.x,
                clipped_destination.y + row_start,
                clipped_destination.w,
                row_end - row_start,
                special_flags,
                -1 if source._alpha is None else source._alpha,
                int(bool(source._flags & SRCALPHA)),
                int(bool(self._flags & SRCALPHA)),
                int(source._colorkey is not None),
                *key,
            )

        parallel_threshold = (
            _PARALLEL_BLIT_PIXELS
            if special_flags in (0, BLEND_PREMULTIPLIED, BLEND_ALPHA_SDL2)
            else _PARALLEL_BLEND_PIXELS
        )
        if clipped_destination.w * clipped_destination.h >= parallel_threshold:
            _parallel_ranges(clipped_destination.h, blit_rows)
        else:
            blit_rows(0, clipped_destination.h)
        return clipped_destination

    def blits(self, blit_sequence, doreturn=1):
        rectangles = [self.blit(*entry) for entry in blit_sequence]
        return rectangles if doreturn else None

    def scroll(self, dx=0, dy=0):
        dx, dy = int(dx), int(dy)
        source = self._pixels.copy()
        x0, x1 = max(0, dx), min(self.get_width(), self.get_width() + dx)
        y0, y1 = max(0, dy), min(self.get_height(), self.get_height() + dy)
        if x1 > x0 and y1 > y0:
            self._pixels[y0:y1, x0:x1] = source[y0 - dy : y1 - dy, x0 - dx : x1 - dx]

    def subsurface(self, rect):
        rect = Rect(rect)
        if not Rect(0, 0, *self.get_size()).contains(rect):
            raise ValueError("subsurface rectangle outside surface area")
        result = Surface.__new__(Surface)
        result._pixels = self._pixels[rect.y : rect.bottom, rect.x : rect.right]
        result._flags = self._flags
        result._alpha = self._alpha
        result._colorkey = self._colorkey
        result._clip = Rect(0, 0, rect.w, rect.h)
        return result


def from_array(array, flags=SRCALPHA):
    return Surface._from_pixels(array, flags)
