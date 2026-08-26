from __future__ import annotations

import math

import numpy as np

from . import _lib
from .mask import from_surface
from .rect import Rect


class Sprite:
    def __init__(self, *groups):
        self.__g = set()
        if groups:
            self.add(*groups)

    def add(self, *groups):
        for group in groups:
            if isinstance(group, (tuple, list, set)):
                self.add(*group)
            elif self not in group:
                group.add(self)

    def remove(self, *groups):
        for group in groups:
            if isinstance(group, (tuple, list, set)):
                self.remove(*group)
            elif self in group:
                group.remove(self)

    def add_internal(self, group):
        self.__g.add(group)

    def remove_internal(self, group):
        self.__g.discard(group)

    def update(self, *args, **kwargs):
        pass

    def kill(self):
        for group in tuple(self.__g):
            group.remove(self)

    def groups(self):
        return list(self.__g)

    def alive(self):
        return bool(self.__g)


class Group:
    def __init__(self, *sprites):
        self.spritedict = {}
        self.lostsprites = []
        if sprites:
            self.add(*sprites)

    def sprites(self):
        return list(self.spritedict)

    def add_internal(self, sprite):
        self.spritedict[sprite] = None

    def remove_internal(self, sprite):
        self.spritedict.pop(sprite, None)

    def has_internal(self, sprite):
        return sprite in self.spritedict

    def add(self, *sprites):
        for sprite in sprites:
            if isinstance(sprite, (tuple, list, set, Group)):
                self.add(*list(sprite))
            elif sprite not in self.spritedict:
                self.add_internal(sprite)
                sprite.add_internal(self)

    def remove(self, *sprites):
        for sprite in sprites:
            if isinstance(sprite, (tuple, list, set, Group)):
                self.remove(*list(sprite))
            elif sprite in self.spritedict:
                self.remove_internal(sprite)
                sprite.remove_internal(self)

    def has(self, *sprites):
        values = []
        for sprite in sprites:
            if isinstance(sprite, (tuple, list, set, Group)):
                values.extend(list(sprite))
            else:
                values.append(sprite)
        return all(sprite in self.spritedict for sprite in values)

    def update(self, *args, **kwargs):
        for sprite in self.sprites():
            sprite.update(*args, **kwargs)

    def draw(self, surface, special_flags=0):
        dirty = []
        for sprite in self.sprites():
            rect = surface.blit(sprite.image, sprite.rect, special_flags=special_flags)
            self.spritedict[sprite] = rect
            dirty.append(rect)
        return dirty

    def clear(self, surface, background):
        for rect in self.spritedict.values():
            if rect is not None:
                surface.blit(background, rect, rect)

    def empty(self):
        self.remove(*self.sprites())

    def copy(self):
        return self.__class__(*self.sprites())

    def __contains__(self, sprite):
        return sprite in self.spritedict

    def __iter__(self):
        return iter(self.sprites())

    def __len__(self):
        return len(self.spritedict)

    def __bool__(self):
        return bool(self.spritedict)


RenderPlain = Group
AbstractGroup = Group
GroupSingle = None


class _GroupSingle(Group):
    @property
    def sprite(self):
        return next(iter(self.spritedict), None)

    @sprite.setter
    def sprite(self, value):
        self.empty()
        if value is not None:
            self.add(value)

    def add(self, *sprites):
        values = []
        for sprite in sprites:
            values.extend(list(sprite) if isinstance(sprite, (tuple, list, set, Group)) else [sprite])
        if values:
            self.empty()
            super().add(values[-1])


GroupSingle = _GroupSingle


def _rect_array(sprites):
    return np.ascontiguousarray(
        [[sprite.rect.x, sprite.rect.y, sprite.rect.w, sprite.rect.h] for sprite in sprites],
        dtype=np.int64,
    )


def _rect_hits(sprite, sprites):
    if not sprites:
        return np.empty(0, dtype=bool)
    target = np.ascontiguousarray(tuple(Rect(sprite.rect)), dtype=np.int64)
    rects = _rect_array(sprites)
    hits = np.empty(len(sprites), dtype=np.uint8)
    _lib.lib().mpg_rect_collisions(
        _lib.addr(target), _lib.addr(rects), len(sprites), _lib.addr(hits)
    )
    return hits.astype(bool, copy=False)


def _colliding_sprites(sprite, sprites):
    target = sprite.rect
    if target.w <= 0 or target.h <= 0:
        return []
    left, top = target.x, target.y
    right, bottom = left + target.w, top + target.h
    result = []
    append = result.append
    for item in sprites:
        rect = item.rect
        rw = rect.w
        rh = rect.h
        rx = rect.x
        ry = rect.y
        if (
            rw > 0
            and rh > 0
            and rx < right
            and rx + rw > left
            and ry < bottom
            and ry + rh > top
        ):
            append(item)
    return result


def collide_rect(left, right):
    return Rect(left.rect).colliderect(right.rect)


class collide_rect_ratio:
    def __init__(self, ratio):
        self.ratio = float(ratio)

    def __call__(self, left, right):
        return Rect(left.rect).inflate(
            int(left.rect.width * self.ratio) - left.rect.width,
            int(left.rect.height * self.ratio) - left.rect.height,
        ).colliderect(
            Rect(right.rect).inflate(
                int(right.rect.width * self.ratio) - right.rect.width,
                int(right.rect.height * self.ratio) - right.rect.height,
            )
        )


def _radius(sprite):
    return float(getattr(sprite, "radius", max(sprite.rect.width, sprite.rect.height) / 2.0))


def collide_circle(left, right):
    dx = left.rect.centerx - right.rect.centerx
    dy = left.rect.centery - right.rect.centery
    radius = _radius(left) + _radius(right)
    return dx * dx + dy * dy <= radius * radius


class collide_circle_ratio:
    def __init__(self, ratio):
        self.ratio = float(ratio)

    def __call__(self, left, right):
        dx = left.rect.centerx - right.rect.centerx
        dy = left.rect.centery - right.rect.centery
        radius = (_radius(left) + _radius(right)) * self.ratio
        return dx * dx + dy * dy <= radius * radius


def collide_mask(left, right):
    left_mask = getattr(left, "mask", None) or from_surface(left.image)
    right_mask = getattr(right, "mask", None) or from_surface(right.image)
    offset = right.rect.x - left.rect.x, right.rect.y - left.rect.y
    return left_mask.overlap(right_mask, offset)


def spritecollide(sprite, group, dokill, collided=None):
    sprites = group.spritedict
    if collided is None or collided is collide_rect:
        result = _colliding_sprites(sprite, sprites)
    else:
        result = [item for item in sprites if collided(sprite, item)]
    if dokill:
        for item in result:
            item.kill()
    return result


def spritecollideany(sprite, group, collided=None):
    sprites = group.spritedict
    if collided is None or collided is collide_rect:
        hits = _colliding_sprites(sprite, sprites)
        return hits[0] if hits else None
    return next((item for item in sprites if collided(sprite, item)), None)


def groupcollide(groupa, groupb, dokilla, dokillb, collided=None):
    sprites_a, sprites_b = groupa.sprites(), groupb.sprites()
    found = {}
    if not sprites_a or not sprites_b:
        return found
    if collided is None or collided is collide_rect:
        rects_a, rects_b = _rect_array(sprites_a), _rect_array(sprites_b)
        hits = np.empty((len(sprites_a), len(sprites_b)), dtype=np.uint8)
        _lib.lib().mpg_group_collisions(
            _lib.addr(rects_a),
            len(sprites_a),
            _lib.addr(rects_b),
            len(sprites_b),
            _lib.addr(hits),
        )
        available = np.ones(len(sprites_b), dtype=bool)
        for i, item in enumerate(sprites_a):
            indices = np.flatnonzero(hits[i].astype(bool) & available)
            matches = [sprites_b[j] for j in indices]
            if matches:
                found[item] = matches
                if dokillb:
                    available[indices] = False
                    for match in matches:
                        match.kill()
    else:
        for item_a in sprites_a:
            matches = [item_b for item_b in sprites_b if collided(item_a, item_b)]
            if matches:
                found[item_a] = matches
    if dokilla:
        for item in found:
            item.kill()
    if dokillb and not (collided is None or collided is collide_rect):
        for matches in found.values():
            for item in matches:
                item.kill()
    return found
