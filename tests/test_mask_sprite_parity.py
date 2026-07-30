import numpy as np
import pygame
import pytest

import mojopygame as mpg

rng = np.random.default_rng(99)


def masks(bits):
    height, width = bits.shape
    ours = mpg.mask.Mask((width, height))
    ours.bits[...] = bits
    theirs = pygame.Mask((width, height))
    for y, x in np.argwhere(bits):
        theirs.set_at((int(x), int(y)))
    return ours, theirs


@pytest.mark.parametrize("offset", [(0, 0), (7, 4), (-9, 3), (5, -11), (100, 100)])
def test_mask_overlap_matches_upstream(offset):
    a = (rng.random((43, 57)) < 0.22).astype(np.uint8)
    b = (rng.random((31, 49)) < 0.28).astype(np.uint8)
    ours_a, theirs_a = masks(a)
    ours_b, theirs_b = masks(b)
    assert ours_a.overlap(ours_b, offset) == theirs_a.overlap(theirs_b, offset)
    assert ours_a.overlap_area(ours_b, offset) == theirs_a.overlap_area(theirs_b, offset)
    assert ours_a.overlap_mask(ours_b, offset).count() == theirs_a.overlap_mask(theirs_b, offset).count()


def test_mask_from_surface_threshold_matches():
    alpha = rng.integers(0, 256, (29, 37), dtype=np.uint8)
    pixels = np.zeros((29, 37, 4), dtype=np.uint8)
    pixels[..., 3] = alpha
    ours_surface = mpg.from_array(pixels)
    theirs_surface = pygame.Surface((37, 29), pygame.SRCALPHA, 32)
    pygame.surfarray.pixels_alpha(theirs_surface)[...] = alpha.T
    for threshold in (0, 63, 127, 200, 255):
        assert mpg.mask.from_surface(ours_surface, threshold).count() == pygame.mask.from_surface(theirs_surface, threshold).count()


def test_mask_overlap_simd_tail_matches_upstream():
    a = (rng.random((11, 35)) < 0.4).astype(np.uint8)
    b = (rng.random((9, 33)) < 0.3).astype(np.uint8)
    ours_a, theirs_a = masks(a)
    ours_b, theirs_b = masks(b)
    offset = (1, -2)
    assert ours_a.overlap(ours_b, offset) == theirs_a.overlap(theirs_b, offset)
    assert ours_a.overlap_area(ours_b, offset) == theirs_a.overlap_area(theirs_b, offset)


def test_mask_mutation_and_overlap_mask_match_upstream():
    ours = mpg.mask.Mask((13, 9))
    theirs = pygame.Mask((13, 9))
    ours_other = mpg.mask.Mask((7, 6), fill=True)
    theirs_other = pygame.Mask((7, 6), fill=True)
    ours.draw(ours_other, (4, 2))
    theirs.draw(theirs_other, (4, 2))
    ours.erase(ours_other, (7, 4))
    theirs.erase(theirs_other, (7, 4))
    ours.invert()
    theirs.invert()
    assert ours.count() == theirs.count()
    ours_overlap = ours.overlap_mask(ours_other, (2, 1))
    theirs_overlap = theirs.overlap_mask(theirs_other, (2, 1))
    assert ours_overlap.count() == theirs_overlap.count()
    assert np.array_equal(
        ours_overlap.bits,
        np.array(
            [
                [theirs_overlap.get_at((x, y)) for x in range(13)]
                for y in range(9)
            ],
            dtype=np.uint8,
        ),
    )


class OursSprite(mpg.sprite.Sprite):
    def __init__(self, rect):
        super().__init__()
        self.rect = mpg.Rect(rect)


class TheirSprite(pygame.sprite.Sprite):
    def __init__(self, rect):
        super().__init__()
        self.rect = pygame.Rect(rect)


def sprite_sets(count=300):
    rects = np.column_stack(
        (
            rng.integers(-500, 500, count),
            rng.integers(-500, 500, count),
            rng.integers(1, 80, count),
            rng.integers(1, 80, count),
        )
    )
    return [OursSprite(r) for r in rects], [TheirSprite(r) for r in rects]


def test_spritecollide_and_any_match_upstream():
    ours, theirs = sprite_sets()
    ours_group, theirs_group = mpg.sprite.Group(ours), pygame.sprite.Group(theirs)
    ours_target, theirs_target = OursSprite((-30, 20, 180, 140)), TheirSprite((-30, 20, 180, 140))
    ours_hits = mpg.sprite.spritecollide(ours_target, ours_group, False)
    theirs_hits = pygame.sprite.spritecollide(theirs_target, theirs_group, False)
    assert [ours.index(item) for item in ours_hits] == [theirs.index(item) for item in theirs_hits]
    ours_any = mpg.sprite.spritecollideany(ours_target, ours_group)
    theirs_any = pygame.sprite.spritecollideany(theirs_target, theirs_group)
    assert ours.index(ours_any) == theirs.index(theirs_any)


def test_groupcollide_and_kill_match_upstream():
    ours_a, theirs_a = sprite_sets(90)
    ours_b, theirs_b = sprite_sets(110)
    oga, ogb = mpg.sprite.Group(ours_a), mpg.sprite.Group(ours_b)
    tga, tgb = pygame.sprite.Group(theirs_a), pygame.sprite.Group(theirs_b)
    ours_found = mpg.sprite.groupcollide(oga, ogb, True, True)
    theirs_found = pygame.sprite.groupcollide(tga, tgb, True, True)
    ours_pairs = {
        ours_a.index(key): [ours_b.index(value) for value in values]
        for key, values in ours_found.items()
    }
    theirs_pairs = {
        theirs_a.index(key): [theirs_b.index(value) for value in values]
        for key, values in theirs_found.items()
    }
    assert ours_pairs == theirs_pairs
    assert len(oga) == len(tga)
    assert len(ogb) == len(tgb)


def test_collision_callbacks_match():
    left_o, right_o = OursSprite((3, 5, 30, 20)), OursSprite((28, 15, 20, 30))
    left_t, right_t = TheirSprite((3, 5, 30, 20)), TheirSprite((28, 15, 20, 30))
    for ratio in (0.5, 1.0, 1.6):
        assert mpg.sprite.collide_rect_ratio(ratio)(left_o, right_o) == pygame.sprite.collide_rect_ratio(ratio)(left_t, right_t)
        assert mpg.sprite.collide_circle_ratio(ratio)(left_o, right_o) == pygame.sprite.collide_circle_ratio(ratio)(left_t, right_t)


def test_collide_mask_returns_same_point():
    a = np.zeros((20, 30, 4), dtype=np.uint8)
    b = np.zeros((17, 25, 4), dtype=np.uint8)
    a[4:12, 8:18, 3] = 255
    b[2:14, 1:8, 3] = 255
    left_o, right_o = OursSprite((10, 20, 30, 20)), OursSprite((23, 25, 25, 17))
    left_o.image, right_o.image = mpg.from_array(a), mpg.from_array(b)
    left_t, right_t = TheirSprite((10, 20, 30, 20)), TheirSprite((23, 25, 25, 17))
    left_t.image = pygame.Surface((30, 20), pygame.SRCALPHA, 32)
    right_t.image = pygame.Surface((25, 17), pygame.SRCALPHA, 32)
    pygame.surfarray.pixels_alpha(left_t.image)[...] = a[..., 3].T
    pygame.surfarray.pixels_alpha(right_t.image)[...] = b[..., 3].T
    assert mpg.sprite.collide_mask(left_o, right_o) == pygame.sprite.collide_mask(left_t, right_t)
