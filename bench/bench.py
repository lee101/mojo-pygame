"""mojopygame against pygame 2.6.1 and NumPy on the same inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import numpy as np  # noqa: E402
import pygame  # noqa: E402

import mojopygame as mpg  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def pygame_surface(pixels):
    height, width = pixels.shape[:2]
    surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
    pygame.surfarray.pixels3d(surface)[...] = pixels[..., :3].transpose(1, 0, 2)
    pygame.surfarray.pixels_alpha(surface)[...] = pixels[..., 3].T
    return surface


class MojoSprite(mpg.sprite.Sprite):
    def __init__(self, rect):
        super().__init__()
        self.rect = mpg.Rect(rect)


class PygameSprite(pygame.sprite.Sprite):
    def __init__(self, rect):
        super().__init__()
        self.rect = pygame.Rect(rect)


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    pygame.init()
    mpg.mixer.init(frequency=48000, size=-16, channels=2)
    rng = np.random.default_rng(2026)
    rows = []

    source_pixels = rng.integers(0, 256, (1080, 1920, 4), dtype=np.uint8)
    dest_pixels = rng.integers(0, 256, (1080, 1920, 4), dtype=np.uint8)
    mojo_source, mojo_dest = mpg.from_array(source_pixels), mpg.from_array(dest_pixels)
    pygame_source, pygame_dest = pygame_surface(source_pixels), pygame_surface(dest_pixels)

    mojo_dest.blit(mojo_source, (0, 0))
    pygame_dest.blit(pygame_source, (0, 0))
    ours = timeit(lambda: mojo_dest.blit(mojo_source, (0, 0)))
    theirs = timeit(lambda: pygame_dest.blit(pygame_source, (0, 0)))
    rows.append(("alpha blit 1920x1080", ours, theirs, "pygame"))

    ours = timeit(
        lambda: mojo_dest.blit(
            mojo_source, (0, 0), special_flags=mpg.BLEND_RGBA_ADD
        )
    )
    theirs = timeit(
        lambda: pygame_dest.blit(
            pygame_source, (0, 0), special_flags=pygame.BLEND_RGBA_ADD
        )
    )
    rows.append(("RGBA add blit 1920x1080", ours, theirs, "pygame"))

    rects = np.column_stack(
        (
            rng.integers(-10000, 10000, 100_000),
            rng.integers(-10000, 10000, 100_000),
            rng.integers(4, 100, 100_000),
            rng.integers(4, 100, 100_000),
        )
    )
    mojo_sprites = [MojoSprite(rect) for rect in rects]
    pygame_sprites = [PygameSprite(rect) for rect in rects]
    mojo_group = mpg.sprite.Group(mojo_sprites)
    pygame_group = pygame.sprite.Group(pygame_sprites)
    mojo_target = MojoSprite((-500, -500, 1000, 1000))
    pygame_target = PygameSprite((-500, -500, 1000, 1000))
    ours = timeit(lambda: mpg.sprite.spritecollide(mojo_target, mojo_group, False))
    theirs = timeit(
        lambda: pygame.sprite.spritecollide(pygame_target, pygame_group, False)
    )
    rows.append(("spritecollide 100k rects", ours, theirs, "pygame"))

    rects_a, rects_b = rects[:1500], rects[1500:3000]
    mga = mpg.sprite.Group([MojoSprite(rect) for rect in rects_a])
    mgb = mpg.sprite.Group([MojoSprite(rect) for rect in rects_b])
    pga = pygame.sprite.Group([PygameSprite(rect) for rect in rects_a])
    pgb = pygame.sprite.Group([PygameSprite(rect) for rect in rects_b])
    ours = timeit(lambda: mpg.sprite.groupcollide(mga, mgb, False, False), repeat=3)
    theirs = timeit(
        lambda: pygame.sprite.groupcollide(pga, pgb, False, False), repeat=3
    )
    rows.append(("groupcollide 1500x1500", ours, theirs, "pygame"))

    bits_a = (rng.random((2048, 2048)) < 0.2).astype(np.uint8)
    bits_b = (rng.random((2048, 2048)) < 0.2).astype(np.uint8)
    mask_a, mask_b = mpg.mask.Mask((2048, 2048)), mpg.mask.Mask((2048, 2048))
    mask_a.bits[...] = bits_a
    mask_b.bits[...] = bits_b
    pygame_a = pygame.mask.from_surface(pygame_surface(np.dstack((
        np.zeros_like(bits_a), np.zeros_like(bits_a), np.zeros_like(bits_a),
        bits_a * 255,
    ))), 127)
    pygame_b = pygame.mask.from_surface(pygame_surface(np.dstack((
        np.zeros_like(bits_b), np.zeros_like(bits_b), np.zeros_like(bits_b),
        bits_b * 255,
    ))), 127)
    ours = timeit(lambda: mask_a.overlap_area(mask_b, (13, -7)))
    theirs = timeit(lambda: pygame_a.overlap_area(pygame_b, (13, -7)))
    rows.append(("mask overlap_area 2048x2048", ours, theirs, "pygame"))

    voices = [
        rng.integers(-4000, 4000, (96_000, 2), dtype=np.int16)
        for _ in range(24)
    ]

    def numpy_mix():
        summed = np.sum(np.stack(voices, axis=0), axis=0, dtype=np.int32)
        return np.clip(summed, -32768, 32767).astype(np.int16)

    mpg.mixer.mix(voices)
    ours = timeit(lambda: mpg.mixer.mix(voices), repeat=3)
    theirs = timeit(numpy_mix, repeat=3)
    rows.append(("mix 24 stereo voices, 2 sec", ours, theirs, "NumPy reference"))

    pcm = rng.integers(-20000, 20000, (480_000, 2), dtype=np.int16)
    target_frames = round(len(pcm) * 44100 / 48000)
    positions = np.arange(target_frames) * 48000 / 44100

    def numpy_resample():
        left = np.minimum(positions.astype(np.int64), len(pcm) - 1)
        right = np.minimum(left + 1, len(pcm) - 1)
        fraction = (positions - left)[:, None]
        source = pcm.astype(np.float64)
        return (
            source[left] + (source[right] - source[left]) * fraction
        ).astype(np.int16)

    ours = timeit(lambda: mpg.mixer.resample(pcm, 48000, 44100), repeat=3)
    theirs = timeit(numpy_resample, repeat=3)
    rows.append(("linear resample 10 sec stereo", ours, theirs, "NumPy reference"))

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print(f"Versions: mojopygame {mpg.__version__}, pygame {pygame.version.ver}, NumPy {np.__version__}")
    print()
    print("| case | mojopygame | reference | ratio | result |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, ours, theirs, reference in rows:
        ratio = theirs / ours
        result = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {ours * 1000:.2f} ms | {theirs * 1000:.2f} ms "
            f"| {ratio:.2f}x | {result} vs {reference} |"
        )


if __name__ == "__main__":
    main()
