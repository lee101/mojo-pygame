# mojo-pygame

`mojo-pygame` is a focused port of pygame's compute-heavy 2D primitives to
[Mojo](https://www.modular.com/mojo). It provides a pygame-shaped Python API
for software surfaces, sprite collision, masks, and deterministic PCM mixing.
NumPy owns the buffers; a single shared Mojo library performs the pixel,
pairwise-collision, mask, mixing, and resampling loops.

The intended migration is an import change for the covered subset:

```python
import mojopygame as pygame
import numpy as np

canvas = pygame.Surface((640, 360), pygame.SRCALPHA)
tile = pygame.Surface((64, 64), pygame.SRCALPHA)
tile.fill((30, 120, 220, 180))
changed = canvas.blit(tile, (24, 32), special_flags=pygame.BLEND_RGBA_ADD)

pygame.mixer.init(frequency=48000, size=-16, channels=2)
tone = pygame.mixer.Sound(array=np.zeros((4800, 2), dtype=np.int16))
tone.play()
pcm = pygame.mixer.render(4800)
```

`changed` is a pygame-style `Rect`; `pcm` is an interleaved stereo `int16`
array ready for an audio device or WAV writer.

## Tested coverage

This is a compatibility subset, not a replacement for the pygame runtime. The
following public behavior is exercised against pygame in the test suite:

| pygame area | covered and tested |
| --- | --- |
| geometry | `Rect` construction and properties; move, inflate, normalize, clip, union, containment, and point/list/rect collision |
| surfaces | 32-bit `Surface`; `from_array`, `blit`, `fill`, clipping, alpha, colorkey, self-blit, strided subsurfaces, pixel access, and copy metadata |
| blending | default and global alpha, opaque surfaces, colorkeys, the exported `BLEND_*` flags, premultiplied alpha, and SDL2 alpha |
| arrays and masks | live `pixels3d` and `pixels_alpha` views; mask overlap point/area/mask, draw, erase, invert, count, and `from_surface` |
| sprites | `Sprite` and `Group` collision behavior; `spritecollide`, `spritecollideany`, `groupcollide`, and rect/circle/ratio/mask callbacks |
| mixer | signed 16-bit stereo array/buffer `Sound`; channel state, queues, loops, volume, offline `mix`, `render`, and linear `resample` |

The parity suite compares against the locked pygame dependency, not a
hand-written substitute.
Default alpha and integer blend modes are pixel exact. SDL's selected
colorkey, premultiplied, and SDL2 alpha implementations can choose
architecture-specific rounding; those paths are required to agree within one
8-bit level while their alpha/key behavior agrees.

The mixer implementation also accepts mono arrays when initialized in mono
mode, but the parity gate currently exercises stereo. Anything not listed
above should be treated as outside the compatibility guarantee.

The package does not open windows or
audio devices and does not implement events, input, display, drawing
primitives, fonts, image codecs, transforms, cameras, MIDI, compressed audio,
unsigned, 8-bit, or float PCM, or pygame's specialized dirty/layered sprite
groups. Virtual
mixer playback advances only when `mixer.render(frames)` is called, making it
deterministic and suitable for engines that already own their audio callback.

## Install

```bash
pixi install
pixi run build
pixi run test
```

Pixi installs the locked Mojo toolchain, Python dependencies, pygame for
parity tests, and NumPy. The build task produces
`dist/libmojo-pygame.so`. The Python package also rebuilds a missing or stale
library on first use. `MOJOPYGAME_LIB=/path/to/libmojo-pygame.so` selects a
prebuilt library.

Run the benchmarks only through the locked task:

```bash
pixi run bench
```

## Performance

Measured by `pixi run bench` on this machine. Times are the best of repeated
warm runs. The pygame
rows use the locked pygame package; mixer/resampling rows use the listed NumPy reference
because pygame does not expose its device mix loop as an offline array
operation.

| case | mojopygame | reference | ratio | result |
| --- | ---: | ---: | ---: | --- |
| alpha blit 1920x1080 | 2.84 ms | 5.33 ms | 1.88x | faster vs pygame |
| RGBA add blit 1920x1080 | 0.92 ms | 0.91 ms | 0.98x | slower vs pygame |
| spritecollide 100k rects | 17.77 ms | 7.55 ms | 0.42x | slower vs pygame |
| groupcollide 1500x1500 | 26.01 ms | 125.80 ms | 4.84x | faster vs pygame |
| mask overlap_area 2048x2048 | 0.52 ms | 0.13 ms | 0.25x | slower vs pygame |
| mix 24 stereo voices, 2 sec | 2.10 ms | 3.91 ms | 1.86x | faster vs NumPy reference |
| linear resample 10 sec stereo | 3.79 ms | 122.67 ms | 32.39x | faster vs NumPy reference |

The byte-mask representation trades pygame's compact bit packing for a simple
NumPy-compatible layout. All covered kernels are below roughly two operations
per byte moved, including blending, mask intersection, mixing, and resampling.
Host/device transfer and launch overhead therefore make them poor GPU targets;
the package intentionally remains CPU-only and has no GPU dependency.

## How it works

```
Python Surface / Group / Sound
        |
        | ctypes: addresses, dimensions, scalar options
        v
src/capi.mojo
        |
        v
src/kernels.mojo
```

Images are row-major `uint8` RGBA with shape `(height, width, 4)`. Subsurfaces
carry an explicit row stride across the ABI. Masks are row-major byte maps.
PCM is C-contiguous
`int16` with shape `(frames, channels)`. Python clips rectangles and owns all
allocation and lifetimes; Mojo receives integer addresses and never retains
them or allocates result buffers. A persistent bounded host worker pool divides
only large independent row or sample ranges; smaller calls stay serial to avoid
thread scheduling overhead.

The C ABI uses `@export("name")` and `abi("C")`. Buffers cross as `Int`
addresses because an exported Mojo function cannot be parametric over pointer
origins. Each wrapper reconstructs an
`UnsafePointer[..., AnyOrigin[mut=True]]` for the duration of one call. One
compilation unit produces one shared library, so import-time ctypes setup and
per-operation crossing costs stay small.

## License

MIT
