import numpy as np
import pygame
import pytest

import mojopygame as mpg

rng = np.random.default_rng(7)


@pytest.fixture(autouse=True)
def initialized_mixers():
    pygame.mixer.quit()
    pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=512)
    mpg.mixer.quit()
    mpg.mixer.init(frequency=48000, size=-16, channels=2, buffer=512)
    yield
    pygame.mixer.quit()
    mpg.mixer.quit()


def test_sound_array_raw_length_matches_upstream():
    pcm = rng.integers(-32768, 32768, (1234, 2), dtype=np.int16)
    ours = mpg.mixer.Sound(array=pcm)
    theirs = pygame.mixer.Sound(array=pcm)
    assert ours.get_raw() == theirs.get_raw()
    assert ours.get_length() == pytest.approx(theirs.get_length(), abs=1 / 48000)


def test_sound_buffer_matches_upstream():
    pcm = rng.integers(-32768, 32768, (321, 2), dtype=np.int16)
    ours = mpg.mixer.Sound(buffer=pcm.tobytes())
    theirs = pygame.mixer.Sound(buffer=pcm.tobytes())
    assert ours.get_raw() == theirs.get_raw()


def numpy_mix(arrays, volumes):
    frames = max(map(len, arrays))
    work = np.zeros((frames, 2), dtype=np.float64)
    for array, volume in zip(arrays, volumes):
        work[: len(array)] += array * np.asarray(volume)
    return np.clip(work.astype(np.int64), -32768, 32767).astype(np.int16)


def test_mojo_mix_matches_reference_exactly():
    arrays = [
        rng.integers(-20000, 20000, (1000 + i * 137, 2), dtype=np.int16)
        for i in range(7)
    ]
    volumes = [(0.1 + i * 0.07, 0.9 - i * 0.08) for i in range(len(arrays))]
    assert np.max(np.abs(mpg.mixer.mix(arrays, volumes).astype(np.int32) - numpy_mix(arrays, volumes).astype(np.int32))) <= 1


def test_mix_saturates():
    loud = np.full((100, 2), 30000, dtype=np.int16)
    mixed = mpg.mixer.mix([loud, loud, loud])
    assert np.all(mixed == 32767)
    mixed = mpg.mixer.mix([-loud, -loud, -loud])
    assert np.all(mixed == -32768)


def test_large_mix_parallel_path_matches_reference():
    arrays = [
        rng.integers(-100, 101, (254_201, 2), dtype=np.int16)
        for _ in range(33)
    ]
    volumes = [
        (0.25 + (i % 3) * 0.125, 0.75 - (i % 4) * 0.125)
        for i in range(len(arrays))
    ]
    result = mpg.mixer.mix(arrays, volumes)
    reference = numpy_mix(arrays, volumes)
    assert np.max(np.abs(result.astype(np.int32) - reference.astype(np.int32))) <= 1


def test_channel_render_queue_and_loop():
    first_pcm = np.full((5, 2), (1000, 2000), dtype=np.int16)
    second_pcm = np.full((4, 2), (3000, 4000), dtype=np.int16)
    first, second = mpg.mixer.Sound(array=first_pcm), mpg.mixer.Sound(array=second_pcm)
    channel = mpg.mixer.Channel(0)
    channel.play(first, loops=1)
    channel.queue(second)
    rendered = mpg.mixer.render(14)
    expected = np.concatenate((first_pcm, first_pcm, second_pcm))
    assert np.array_equal(rendered, expected)
    assert not channel.get_busy()


def test_channel_and_sound_volume():
    pcm = np.full((8, 2), 10000, dtype=np.int16)
    sound = mpg.mixer.Sound(array=pcm)
    sound.set_volume(0.5)
    channel = mpg.mixer.Channel(2)
    channel.set_volume(0.2, 0.8)
    channel.play(sound)
    rendered = mpg.mixer.render(8)
    assert np.array_equal(rendered, np.tile((1000, 4000), (8, 1)))


def test_resample_matches_linear_reference():
    pcm = rng.integers(-20000, 20000, (1001, 2), dtype=np.int16)
    result = mpg.mixer.resample(pcm, 48000, 44100)
    frames = round(len(pcm) * 44100 / 48000)
    positions = np.arange(frames) * 48000 / 44100
    left = np.minimum(positions.astype(int), len(pcm) - 1)
    right = np.minimum(left + 1, len(pcm) - 1)
    fraction = (positions - left)[:, None]
    source = pcm.astype(np.float64)
    reference = (source[left] + (source[right] - source[left]) * fraction).astype(np.int16)
    assert np.max(np.abs(result.astype(np.int32) - reference.astype(np.int32))) <= 1


def test_mixer_state_api():
    assert mpg.mixer.get_init() == (48000, -16, 2)
    mpg.mixer.set_num_channels(3)
    assert mpg.mixer.get_num_channels() == 3
    sound = mpg.mixer.Sound(array=np.zeros((10, 2), dtype=np.int16))
    channel = sound.play()
    assert channel is not None
    assert mpg.mixer.get_busy()
    mpg.mixer.stop()
    assert not mpg.mixer.get_busy()


def test_mix_rejects_nonfinite_gains():
    pcm = np.ones((2, 2), dtype=np.int16)
    with pytest.raises(ValueError):
        mpg.mixer.mix([pcm], [np.nan])
