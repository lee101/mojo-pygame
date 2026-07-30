import numpy as np
import pygame
import pytest

import mojopygame as mpg

rng = np.random.default_rng(123)


def pygame_surface(pixels, flags=pygame.SRCALPHA):
    height, width = pixels.shape[:2]
    surface = pygame.Surface((width, height), flags, 32)
    pygame.surfarray.pixels3d(surface)[...] = pixels[..., :3].transpose(1, 0, 2)
    if flags & pygame.SRCALPHA:
        pygame.surfarray.pixels_alpha(surface)[...] = pixels[..., 3].T
    return surface


def pygame_pixels(surface):
    rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    alpha = (
        pygame.surfarray.array_alpha(surface).T
        if surface.get_flags() & pygame.SRCALPHA
        else np.full(surface.get_size()[::-1], 255, dtype=np.uint8)
    )
    return np.dstack((rgb, alpha))


@pytest.mark.parametrize(
    "flag",
    [
        0,
        pygame.BLEND_ADD,
        pygame.BLEND_SUB,
        pygame.BLEND_MULT,
        pygame.BLEND_MIN,
        pygame.BLEND_MAX,
        pygame.BLEND_RGBA_ADD,
        pygame.BLEND_RGBA_SUB,
        pygame.BLEND_RGBA_MULT,
        pygame.BLEND_RGBA_MIN,
        pygame.BLEND_RGBA_MAX,
        pygame.BLEND_PREMULTIPLIED,
        pygame.BLEND_ALPHA_SDL2,
    ],
)
def test_blit_modes_are_pixel_exact(flag):
    source_pixels = rng.integers(0, 256, (37, 53, 4), dtype=np.uint8)
    dest_pixels = rng.integers(0, 256, (48, 61, 4), dtype=np.uint8)
    ours_source = mpg.from_array(source_pixels)
    ours_dest = mpg.from_array(dest_pixels.copy())
    theirs_source = pygame_surface(source_pixels)
    theirs_dest = pygame_surface(dest_pixels)
    ours_rect = ours_dest.blit(ours_source, (4, 6), (3, 2, 42, 31), flag)
    theirs_rect = theirs_dest.blit(theirs_source, (4, 6), (3, 2, 42, 31), flag)
    assert tuple(ours_rect) == tuple(theirs_rect)
    difference = np.abs(
        ours_dest.pixels.astype(np.int16) - pygame_pixels(theirs_dest).astype(np.int16)
    )
    tolerance = 1 if flag in (pygame.BLEND_PREMULTIPLIED, pygame.BLEND_ALPHA_SDL2) else 0
    assert difference.max() <= tolerance


@pytest.mark.parametrize("alpha", [0, 1, 63, 127, 128, 200, 255, None])
def test_global_alpha_is_pixel_exact(alpha):
    source_pixels = rng.integers(0, 256, (19, 23, 4), dtype=np.uint8)
    dest_pixels = rng.integers(0, 256, (22, 29, 4), dtype=np.uint8)
    ours_source = mpg.from_array(source_pixels)
    ours_dest = mpg.from_array(dest_pixels.copy())
    theirs_source = pygame_surface(source_pixels)
    theirs_dest = pygame_surface(dest_pixels)
    ours_source.set_alpha(alpha)
    theirs_source.set_alpha(alpha)
    ours_dest.blit(ours_source, (-2, 3))
    theirs_dest.blit(theirs_source, (-2, 3))
    difference = np.abs(
        ours_dest.pixels.astype(np.int16) - pygame_pixels(theirs_dest).astype(np.int16)
    )
    assert difference.max() <= 1


@pytest.mark.parametrize("source_alpha,dest_alpha", [(False, False), (False, True), (True, False), (True, True)])
def test_opaque_and_alpha_surface_rules(source_alpha, dest_alpha):
    source_pixels = rng.integers(0, 256, (13, 17, 4), dtype=np.uint8)
    dest_pixels = rng.integers(0, 256, (15, 20, 4), dtype=np.uint8)
    sf = mpg.SRCALPHA if source_alpha else 0
    df = mpg.SRCALPHA if dest_alpha else 0
    psf = pygame.SRCALPHA if source_alpha else 0
    pdf = pygame.SRCALPHA if dest_alpha else 0
    ours_source = mpg.from_array(source_pixels, sf)
    ours_dest = mpg.from_array(dest_pixels.copy(), df)
    theirs_source = pygame_surface(source_pixels, psf)
    theirs_dest = pygame_surface(dest_pixels, pdf)
    ours_dest.blit(ours_source, (1, 1))
    theirs_dest.blit(theirs_source, (1, 1))
    difference = np.abs(
        ours_dest.pixels.astype(np.int16) - pygame_pixels(theirs_dest).astype(np.int16)
    )
    assert difference.max() <= 1


def test_clipping_area_and_return_rect_match():
    pixels = rng.integers(0, 256, (30, 40, 4), dtype=np.uint8)
    ours_source, theirs_source = mpg.from_array(pixels), pygame_surface(pixels)
    ours_dest, theirs_dest = mpg.Surface((25, 20), mpg.SRCALPHA), pygame.Surface((25, 20), pygame.SRCALPHA, 32)
    ours_dest.set_clip((3, 4, 15, 11))
    theirs_dest.set_clip((3, 4, 15, 11))
    ours_rect = ours_dest.blit(ours_source, (-5, 1), (-3, 2, 35, 24))
    theirs_rect = theirs_dest.blit(theirs_source, (-5, 1), (-3, 2, 35, 24))
    assert tuple(ours_rect) == tuple(theirs_rect)
    assert np.array_equal(ours_dest.pixels, pygame_pixels(theirs_dest))


def test_colorkey_matches_upstream():
    pixels = rng.integers(0, 256, (20, 25, 4), dtype=np.uint8)
    pixels[2::3, 1::4, :3] = (12, 34, 56)
    ours_source, theirs_source = mpg.from_array(pixels), pygame_surface(pixels)
    ours_dest, theirs_dest = mpg.Surface((25, 20), mpg.SRCALPHA), pygame.Surface((25, 20), pygame.SRCALPHA, 32)
    ours_source.set_colorkey((12, 34, 56))
    theirs_source.set_colorkey((12, 34, 56))
    ours_dest.blit(ours_source, (0, 0))
    theirs_dest.blit(theirs_source, (0, 0))
    difference = np.abs(
        ours_dest.pixels.astype(np.int16) - pygame_pixels(theirs_dest).astype(np.int16)
    )
    assert difference.max() <= 1
    assert np.array_equal(ours_dest.pixels[..., 3], pygame_pixels(theirs_dest)[..., 3])


def test_rgba_add_simd_tail_is_pixel_exact():
    source_pixels = rng.integers(0, 256, (9, 35, 4), dtype=np.uint8)
    dest_pixels = rng.integers(0, 256, (9, 35, 4), dtype=np.uint8)
    ours_source, theirs_source = mpg.from_array(source_pixels), pygame_surface(source_pixels)
    ours_dest, theirs_dest = mpg.from_array(dest_pixels.copy()), pygame_surface(dest_pixels)
    ours_dest.blit(ours_source, (0, 0), special_flags=mpg.BLEND_RGBA_ADD)
    theirs_dest.blit(theirs_source, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    assert np.array_equal(ours_dest.pixels, pygame_pixels(theirs_dest))


def test_large_alpha_blit_parallel_path_matches_upstream():
    source_pixels = rng.integers(0, 256, (513, 513, 4), dtype=np.uint8)
    dest_pixels = rng.integers(0, 256, (513, 513, 4), dtype=np.uint8)
    ours_source, theirs_source = mpg.from_array(source_pixels), pygame_surface(source_pixels)
    ours_dest, theirs_dest = mpg.from_array(dest_pixels.copy()), pygame_surface(dest_pixels)
    ours_dest.blit(ours_source, (0, 0))
    theirs_dest.blit(theirs_source, (0, 0))
    assert np.array_equal(ours_dest.pixels, pygame_pixels(theirs_dest))


def test_overlapping_self_blit_matches_upstream():
    pixels = rng.integers(0, 256, (31, 41, 4), dtype=np.uint8)
    ours, theirs = mpg.from_array(pixels), pygame_surface(pixels)
    ours.blit(ours, (7, 5), (1, 2, 29, 23))
    theirs.blit(theirs, (7, 5), (1, 2, 29, 23))
    assert np.array_equal(ours.pixels, pygame_pixels(theirs))


def test_overlapping_sibling_subsurface_blit_is_alias_safe():
    pixels = rng.integers(0, 256, (25, 35, 4), dtype=np.uint8)
    ours_parent, theirs_parent = mpg.from_array(pixels), pygame_surface(pixels)
    ours_source = ours_parent.subsurface((1, 2, 25, 17))
    ours_dest = ours_parent.subsurface((6, 5, 25, 17))
    theirs_source = theirs_parent.subsurface((1, 2, 25, 17)).copy()
    theirs_dest = theirs_parent.subsurface((6, 5, 25, 17))
    ours_dest.blit(ours_source, (3, 2))
    theirs_dest.blit(theirs_source, (3, 2))
    assert np.array_equal(ours_parent.pixels, pygame_pixels(theirs_parent))


@pytest.mark.parametrize("use_subsurface_source", [False, True])
def test_subsurface_blit_uses_parent_row_stride(use_subsurface_source):
    source_pixels = rng.integers(0, 256, (17, 23, 4), dtype=np.uint8)
    dest_pixels = rng.integers(0, 256, (19, 29, 4), dtype=np.uint8)
    ours_source_parent = mpg.from_array(source_pixels)
    ours_dest_parent = mpg.from_array(dest_pixels.copy())
    theirs_source_parent = pygame_surface(source_pixels)
    theirs_dest_parent = pygame_surface(dest_pixels)
    ours_source = (
        ours_source_parent.subsurface((3, 2, 17, 11))
        if use_subsurface_source
        else ours_source_parent
    )
    theirs_source = (
        theirs_source_parent.subsurface((3, 2, 17, 11))
        if use_subsurface_source
        else theirs_source_parent
    )
    ours_dest = ours_dest_parent.subsurface((4, 3, 19, 13))
    theirs_dest = theirs_dest_parent.subsurface((4, 3, 19, 13))
    ours_dest.blit(ours_source, (1, 1), special_flags=mpg.BLEND_RGBA_ADD)
    theirs_dest.blit(theirs_source, (1, 1), special_flags=pygame.BLEND_RGBA_ADD)
    assert np.array_equal(ours_dest_parent.pixels, pygame_pixels(theirs_dest_parent))


@pytest.mark.parametrize("flag", [0, pygame.BLEND_RGBA_ADD, pygame.BLEND_RGBA_MULT])
def test_fill_matches_upstream(flag):
    ours, theirs = mpg.Surface((17, 13), mpg.SRCALPHA), pygame.Surface((17, 13), pygame.SRCALPHA, 32)
    ours.fill((37, 91, 173, 119))
    theirs.fill((37, 91, 173, 119))
    ours_rect = ours.fill((230, 20, 80, 140), (-2, 3, 14, 20), flag)
    theirs_rect = theirs.fill((230, 20, 80, 140), (-2, 3, 14, 20), flag)
    assert tuple(ours_rect) == tuple(theirs_rect)
    assert np.array_equal(ours.pixels, pygame_pixels(theirs))


def test_surfarray_axis_order_and_live_views_match():
    ours = mpg.Surface((7, 5), mpg.SRCALPHA)
    theirs = pygame.Surface((7, 5), pygame.SRCALPHA, 32)
    ours_rgb = mpg.surfarray.pixels3d(ours)
    theirs_rgb = pygame.surfarray.pixels3d(theirs)
    values = rng.integers(0, 256, ours_rgb.shape, dtype=np.uint8)
    ours_rgb[...] = values
    theirs_rgb[...] = values
    ours_alpha = mpg.surfarray.pixels_alpha(ours)
    theirs_alpha = pygame.surfarray.pixels_alpha(theirs)
    ours_alpha[...] = theirs_alpha[...] = 77
    assert np.array_equal(ours.pixels, pygame_pixels(theirs))


def test_surface_metadata_and_copy():
    surface = mpg.Surface((12, 9), mpg.SRCALPHA)
    surface.set_alpha(123)
    surface.set_colorkey((1, 2, 3))
    copied = surface.copy()
    assert copied.get_size() == (12, 9)
    assert copied.get_bitsize() == 32
    assert copied.get_bytesize() == 4
    assert copied.get_pitch() == 48
    assert copied.get_alpha() == 123
    assert copied.get_colorkey() == (1, 2, 3, 255)
    copied.set_at((0, 0), (4, 5, 6, 7))
    assert surface.get_at((0, 0)) != copied.get_at((0, 0))


def test_pixel_array_inputs_reject_silent_narrowing():
    with pytest.raises(ValueError):
        mpg.from_array(np.full((2, 3, 4), 256, dtype=np.uint16))
    with pytest.raises(TypeError):
        mpg.from_array(np.zeros((2, 3, 4), dtype=np.float64))
