from __future__ import annotations

import numpy as np

from . import _lib


class Mask:
    def __init__(self, size, fill=False):
        width, height = (int(v) for v in size)
        if width < 0 or height < 0:
            raise ValueError("negative mask dimensions")
        self._bits = np.full((height, width), bool(fill), dtype=np.uint8)

    @classmethod
    def _from_bits(cls, bits):
        bits = np.ascontiguousarray(bits, dtype=np.uint8)
        result = cls((bits.shape[1], bits.shape[0]))
        result._bits = bits
        return result

    @property
    def bits(self):
        return self._bits

    def get_size(self):
        return self._bits.shape[1], self._bits.shape[0]

    def get_at(self, pos):
        x, y = (int(v) for v in pos)
        return int(self._bits[y, x])

    def set_at(self, pos, value=1):
        x, y = (int(v) for v in pos)
        self._bits[y, x] = bool(value)

    def fill(self):
        self._bits.fill(1)

    def clear(self):
        self._bits.fill(0)

    def invert(self):
        np.logical_not(self._bits, out=self._bits)

    def count(self):
        return int(self._bits.sum())

    def overlap(self, other, offset):
        _, first = self._overlap(other, offset)
        return first

    def overlap_area(self, other, offset):
        count, _ = self._overlap(other, offset)
        return count

    def overlap_mask(self, other, offset):
        width, height = self.get_size()
        result = Mask((width, height))
        if width == 0 or height == 0 or other.get_width() == 0 or other.get_height() == 0:
            return result
        first = np.empty(2, dtype=np.int64)
        ox, oy = (int(v) for v in offset)
        _lib.lib().mpg_mask_overlap_into(
            _lib.addr(self._bits),
            width,
            height,
            _lib.addr(other._bits),
            other.get_width(),
            other.get_height(),
            ox,
            oy,
            _lib.addr(first),
            _lib.addr(result._bits),
        )
        return result

    def _overlap(self, other, offset):
        if not isinstance(other, Mask):
            raise TypeError("other must be a Mask")
        width, height = self.get_size()
        if width == 0 or height == 0 or other.get_width() == 0 or other.get_height() == 0:
            return 0, None
        first = np.empty(2, dtype=np.int64)
        ox, oy = (int(v) for v in offset)
        count = _lib.lib().mpg_mask_overlap(
            _lib.addr(self._bits),
            width,
            height,
            _lib.addr(other._bits),
            other.get_width(),
            other.get_height(),
            ox,
            oy,
            _lib.addr(first),
        )
        point = None if first[0] < 0 else (int(first[0]), int(first[1]))
        return int(count), point

    def get_width(self):
        return self._bits.shape[1]

    def get_height(self):
        return self._bits.shape[0]

    def copy(self):
        return Mask._from_bits(self._bits.copy())

    def draw(self, other, offset=(0, 0)):
        ox, oy = (int(v) for v in offset)
        x0, y0 = max(0, ox), max(0, oy)
        x1 = min(self.get_width(), ox + other.get_width())
        y1 = min(self.get_height(), oy + other.get_height())
        if x1 > x0 and y1 > y0:
            np.bitwise_or(
                self._bits[y0:y1, x0:x1],
                other._bits[y0 - oy : y1 - oy, x0 - ox : x1 - ox],
                out=self._bits[y0:y1, x0:x1],
            )

    def erase(self, other, offset=(0, 0)):
        ox, oy = (int(v) for v in offset)
        x0, y0 = max(0, ox), max(0, oy)
        x1 = min(self.get_width(), ox + other.get_width())
        y1 = min(self.get_height(), oy + other.get_height())
        if x1 > x0 and y1 > y0:
            self._bits[y0:y1, x0:x1] &= ~other._bits[
                y0 - oy : y1 - oy, x0 - ox : x1 - ox
            ]

    def __repr__(self):
        return f"<Mask({self.get_width()}x{self.get_height()})>"


def from_surface(surface, threshold=127):
    if surface.get_colorkey() is not None:
        key = np.asarray(surface.get_colorkey()[:3], dtype=np.uint8)
        bits = np.any(surface.pixels[..., :3] != key, axis=2)
    elif surface.get_flags() & 65536:
        bits = surface.pixels[..., 3] > int(threshold)
    else:
        bits = np.ones(surface.pixels.shape[:2], dtype=bool)
    return Mask._from_bits(np.ascontiguousarray(bits, dtype=np.uint8))


def from_threshold(surface, color, threshold=(0, 0, 0, 255), othersurface=None, palette_colors=1):
    reference = surface.pixels if othersurface is None else othersurface.pixels
    color = np.asarray(tuple(color)[:4], dtype=np.int16)
    if color.size == 3:
        color = np.append(color, 255)
    tolerance = np.asarray(tuple(threshold)[:4], dtype=np.int16)
    pixels = surface.pixels.astype(np.int16)
    if othersurface is None:
        bits = np.all(np.abs(pixels - color) < tolerance, axis=2)
    else:
        bits = np.all(np.abs(pixels - reference.astype(np.int16)) < tolerance, axis=2)
    return Mask._from_bits(np.ascontiguousarray(bits, dtype=np.uint8))
