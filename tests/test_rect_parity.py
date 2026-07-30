import pygame
import pytest

from mojopygame import Rect


@pytest.mark.parametrize(
    "values",
    [(1, 2, 30, 40), (-12, 7, 9, 14), (5, 6, 0, 2), (8, 4, -3, -9)],
)
def test_rect_properties_match(values):
    ours, theirs = Rect(values), pygame.Rect(values)
    for name in (
        "left",
        "right",
        "top",
        "bottom",
        "center",
        "centerx",
        "centery",
        "size",
        "topleft",
        "topright",
        "bottomleft",
        "bottomright",
        "midtop",
        "midbottom",
        "midleft",
        "midright",
    ):
        assert getattr(ours, name) == getattr(theirs, name)


@pytest.mark.parametrize("delta", [(3, 5), (-4, 7), (9, -2)])
def test_move_and_inflate_match(delta):
    ours, theirs = Rect(4, 7, 15, 20), pygame.Rect(4, 7, 15, 20)
    assert tuple(ours.move(delta)) == tuple(theirs.move(delta))
    assert tuple(ours.inflate(delta)) == tuple(theirs.inflate(delta))
    ours.move_ip(delta)
    theirs.move_ip(delta)
    assert tuple(ours) == tuple(theirs)


@pytest.mark.parametrize(
    "left,right",
    [
        ((0, 0, 10, 10), (5, 6, 10, 10)),
        ((0, 0, 10, 10), (10, 0, 3, 3)),
        ((-5, -5, 2, 2), (-9, -9, 3, 3)),
        ((0, 0, 0, 10), (0, 0, 10, 10)),
    ],
)
def test_collision_clip_union_match(left, right):
    ours, theirs = Rect(left), pygame.Rect(left)
    assert ours.colliderect(right) == theirs.colliderect(right)
    assert tuple(ours.clip(right)) == tuple(theirs.clip(right))
    assert tuple(ours.union(right)) == tuple(theirs.union(right))


def test_setters_and_get_rect_style_construction():
    ours, theirs = Rect((1, 2), (30, 40)), pygame.Rect((1, 2), (30, 40))
    for name, value in [
        ("right", 80),
        ("bottom", 90),
        ("center", (20, 30)),
        ("midleft", (4, 18)),
        ("size", (11, 13)),
    ]:
        setattr(ours, name, value)
        setattr(theirs, name, value)
        assert tuple(ours) == tuple(theirs)


def test_point_list_contains_and_normalize_match():
    ours, theirs = Rect(10, 20, -30, -40), pygame.Rect(10, 20, -30, -40)
    ours.normalize()
    theirs.normalize()
    assert tuple(ours) == tuple(theirs)
    points = [(-20, -20), (-19, -19), (9, 19), (10, 20)]
    assert [ours.collidepoint(p) for p in points] == [theirs.collidepoint(p) for p in points]
    rects = [(-100, 0, 2, 2), (-10, -10, 5, 5), (100, 100, 1, 1)]
    assert ours.collidelist(rects) == theirs.collidelist(rects)
    assert ours.collidelistall(rects) == theirs.collidelistall(rects)
