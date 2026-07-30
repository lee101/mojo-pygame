from __future__ import annotations

from operator import index


def _pair(value):
    if hasattr(value, "x") and hasattr(value, "y"):
        return int(value.x), int(value.y)
    x, y = value
    return int(x), int(y)


class Rect:
    __slots__ = ("x", "y", "w", "h")

    def __init__(self, *args):
        if len(args) == 1:
            value = args[0]
            if isinstance(value, Rect):
                values = value.x, value.y, value.w, value.h
            elif hasattr(value, "rect"):
                rect = value.rect
                values = rect.x, rect.y, rect.w, rect.h
            else:
                values = tuple(value)
        elif len(args) == 2:
            values = (*_pair(args[0]), *_pair(args[1]))
        elif len(args) == 4:
            values = args
        else:
            raise TypeError("Argument must be rect style object")
        if len(values) != 4:
            raise TypeError("Argument must be rect style object")
        self.x, self.y, self.w, self.h = (int(v) for v in values)

    @property
    def width(self):
        return self.w

    @width.setter
    def width(self, value):
        self.w = int(value)

    @property
    def height(self):
        return self.h

    @height.setter
    def height(self, value):
        self.h = int(value)

    @property
    def left(self):
        return self.x

    @left.setter
    def left(self, value):
        self.x = int(value)

    @property
    def right(self):
        return self.x + self.w

    @right.setter
    def right(self, value):
        self.x = int(value) - self.w

    @property
    def top(self):
        return self.y

    @top.setter
    def top(self, value):
        self.y = int(value)

    @property
    def bottom(self):
        return self.y + self.h

    @bottom.setter
    def bottom(self, value):
        self.y = int(value) - self.h

    @property
    def centerx(self):
        return self.x + self.w // 2

    @centerx.setter
    def centerx(self, value):
        self.x = int(value) - self.w // 2

    @property
    def centery(self):
        return self.y + self.h // 2

    @centery.setter
    def centery(self, value):
        self.y = int(value) - self.h // 2

    @property
    def center(self):
        return self.centerx, self.centery

    @center.setter
    def center(self, value):
        self.centerx, self.centery = _pair(value)

    @property
    def size(self):
        return self.w, self.h

    @size.setter
    def size(self, value):
        self.w, self.h = _pair(value)

    @property
    def topleft(self):
        return self.left, self.top

    @topleft.setter
    def topleft(self, value):
        self.left, self.top = _pair(value)

    @property
    def topright(self):
        return self.right, self.top

    @topright.setter
    def topright(self, value):
        self.right, self.top = _pair(value)

    @property
    def bottomleft(self):
        return self.left, self.bottom

    @bottomleft.setter
    def bottomleft(self, value):
        self.left, self.bottom = _pair(value)

    @property
    def bottomright(self):
        return self.right, self.bottom

    @bottomright.setter
    def bottomright(self, value):
        self.right, self.bottom = _pair(value)

    @property
    def midtop(self):
        return self.centerx, self.top

    @midtop.setter
    def midtop(self, value):
        self.centerx, self.top = _pair(value)

    @property
    def midbottom(self):
        return self.centerx, self.bottom

    @midbottom.setter
    def midbottom(self, value):
        self.centerx, self.bottom = _pair(value)

    @property
    def midleft(self):
        return self.left, self.centery

    @midleft.setter
    def midleft(self, value):
        self.left, self.centery = _pair(value)

    @property
    def midright(self):
        return self.right, self.centery

    @midright.setter
    def midright(self, value):
        self.right, self.centery = _pair(value)

    def copy(self):
        return Rect(self)

    def move(self, *args):
        dx, dy = _pair(args[0] if len(args) == 1 else args)
        return Rect(self.x + dx, self.y + dy, self.w, self.h)

    def move_ip(self, *args):
        dx, dy = _pair(args[0] if len(args) == 1 else args)
        self.x += dx
        self.y += dy

    def inflate(self, *args):
        dw, dh = _pair(args[0] if len(args) == 1 else args)
        return Rect(self.x - dw // 2, self.y - dh // 2, self.w + dw, self.h + dh)

    def inflate_ip(self, *args):
        value = self.inflate(*args)
        self.x, self.y, self.w, self.h = value

    def normalize(self):
        if self.w < 0:
            self.x += self.w
            self.w = -self.w
        if self.h < 0:
            self.y += self.h
            self.h = -self.h

    def clip(self, other):
        other = Rect(other)
        left, top = max(self.left, other.left), max(self.top, other.top)
        right, bottom = min(self.right, other.right), min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return Rect(self.x, self.y, 0, 0)
        return Rect(left, top, right - left, bottom - top)

    def union(self, other):
        other = Rect(other)
        left, top = min(self.left, other.left), min(self.top, other.top)
        right, bottom = max(self.right, other.right), max(self.bottom, other.bottom)
        return Rect(left, top, right - left, bottom - top)

    def unionall(self, rects):
        result = self.copy()
        for rect in rects:
            result = result.union(rect)
        return result

    def clamp(self, other):
        other = Rect(other)
        result = self.copy()
        if result.w >= other.w:
            result.centerx = other.centerx
        else:
            result.left = min(max(result.left, other.left), other.right - result.w)
        if result.h >= other.h:
            result.centery = other.centery
        else:
            result.top = min(max(result.top, other.top), other.bottom - result.h)
        return result

    def clamp_ip(self, other):
        value = self.clamp(other)
        self.x, self.y = value.x, value.y

    def contains(self, other):
        other = Rect(other)
        return (
            other.w > 0
            and other.h > 0
            and self.left <= other.left
            and self.top <= other.top
            and self.right >= other.right
            and self.bottom >= other.bottom
        )

    def collidepoint(self, *args):
        px, py = _pair(args[0] if len(args) == 1 else args)
        return self.w > 0 and self.h > 0 and self.left <= px < self.right and self.top <= py < self.bottom

    def colliderect(self, other):
        other = Rect(other)
        return (
            self.w > 0
            and self.h > 0
            and other.w > 0
            and other.h > 0
            and self.left < other.right
            and self.right > other.left
            and self.top < other.bottom
            and self.bottom > other.top
        )

    def collidelist(self, rects):
        return next((i for i, rect in enumerate(rects) if self.colliderect(rect)), -1)

    def collidelistall(self, rects):
        return [i for i, rect in enumerate(rects) if self.colliderect(rect)]

    def __iter__(self):
        return iter((self.x, self.y, self.w, self.h))

    def __len__(self):
        return 4

    def __getitem__(self, item):
        return (self.x, self.y, self.w, self.h)[index(item)]

    def __setitem__(self, item, value):
        values = [self.x, self.y, self.w, self.h]
        values[index(item)] = int(value)
        self.x, self.y, self.w, self.h = values

    def __bool__(self):
        return self.w != 0 and self.h != 0

    def __eq__(self, other):
        try:
            return tuple(self) == tuple(Rect(other))
        except (TypeError, ValueError):
            return False

    def __repr__(self):
        return f"<rect({self.x}, {self.y}, {self.w}, {self.h})>"
