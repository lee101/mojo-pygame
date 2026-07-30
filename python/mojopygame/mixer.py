"""pygame.mixer-shaped, device-independent PCM mixing.

Playback state advances when ``render(frames)`` is called. This keeps the
module deterministic and lets applications hand the returned PCM to any audio
device or file writer.
"""

from __future__ import annotations

import os
import wave

import numpy as np

from . import _lib

AUDIO_ALLOW_FREQUENCY_CHANGE = 1
AUDIO_ALLOW_FORMAT_CHANGE = 2
AUDIO_ALLOW_CHANNELS_CHANGE = 4
AUDIO_ALLOW_ANY_CHANGE = 7

_pre_init = (44100, -16, 2, 512)
_config = None
_channel_count = 8
_channels = {}


def pre_init(frequency=44100, size=-16, channels=2, buffer=512, devicename=None, allowedchanges=0):
    global _pre_init
    _pre_init = int(frequency), int(size), int(channels), int(buffer)


def init(frequency=44100, size=-16, channels=2, buffer=512, devicename=None, allowedchanges=AUDIO_ALLOW_FREQUENCY_CHANGE | AUDIO_ALLOW_CHANNELS_CHANGE):
    global _config
    frequency, size, channels, buffer = (
        int(frequency),
        int(size),
        int(channels),
        int(buffer),
    )
    if frequency <= 0 or channels not in (1, 2):
        raise ValueError("frequency must be positive and channels must be 1 or 2")
    if size != -16:
        raise ValueError("mojopygame.mixer currently supports signed 16-bit PCM")
    _config = frequency, size, channels
    set_num_channels(_channel_count)


def quit():
    global _config
    stop()
    _config = None


def get_init():
    return _config


def _require_init():
    if _config is None:
        raise RuntimeError("mixer not initialized")
    return _config


def _coerce_pcm(array):
    _, _, channels = _require_init()
    pcm = np.asarray(array)
    if pcm.dtype != np.int16:
        raise ValueError("array must be signed 16-bit")
    if channels == 1:
        if pcm.ndim == 2 and pcm.shape[1] == 1:
            pcm = pcm[:, 0]
        if pcm.ndim != 1:
            raise ValueError("Array must be 1-dimensional for mono mixer")
        pcm = pcm.reshape(-1, 1)
    elif pcm.ndim != 2 or pcm.shape[1] != channels:
        raise ValueError("Array depth must match number of mixer channels")
    return np.ascontiguousarray(pcm, dtype=np.int16)


class Sound:
    def __init__(self, *args, file=None, buffer=None, array=None):
        _require_init()
        supplied = len(args) + sum(value is not None for value in (file, buffer, array))
        if supplied != 1:
            raise TypeError("Sound takes exactly one of a positional source, file, buffer, or array")
        if args:
            value = args[0]
            if isinstance(value, (str, os.PathLike)) or hasattr(value, "read"):
                file = value
            elif isinstance(value, np.ndarray):
                array = value
            else:
                buffer = value
        if array is not None:
            self._pcm = _coerce_pcm(array).copy()
        elif buffer is not None:
            raw = memoryview(buffer).tobytes()
            if len(raw) % (2 * _config[2]):
                raise ValueError("buffer length is not a whole number of PCM frames")
            self._pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, _config[2]).copy()
        else:
            self._pcm = self._read_wave(file)
        self._volume = 1.0

    @staticmethod
    def _read_wave(source):
        with wave.open(source, "rb") as handle:
            if handle.getsampwidth() != 2:
                raise ValueError("only 16-bit WAV files are supported")
            channels = handle.getnchannels()
            rate = handle.getframerate()
            pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").reshape(-1, channels)
        if channels != _config[2]:
            if channels == 1 and _config[2] == 2:
                pcm = np.repeat(pcm, 2, axis=1)
            elif channels == 2 and _config[2] == 1:
                pcm = (pcm.astype(np.int32).sum(axis=1) // 2).astype(np.int16)[:, None]
            else:
                raise ValueError("unsupported WAV channel count")
        pcm = np.ascontiguousarray(pcm, dtype=np.int16)
        if rate != _config[0]:
            pcm = resample(pcm, rate, _config[0])
        return pcm

    def play(self, loops=0, maxtime=0, fade_ms=0):
        channel = find_channel()
        if channel is not None:
            channel.play(self, loops, maxtime, fade_ms)
        return channel

    def stop(self):
        for channel in _channels.values():
            if channel.get_sound() is self:
                channel.stop()

    def fadeout(self, time):
        self.stop()

    def get_num_channels(self):
        return sum(channel.get_sound() is self for channel in _channels.values())

    def get_length(self):
        return len(self._pcm) / _config[0]

    def get_raw(self):
        data = self._pcm[:, 0] if _config[2] == 1 else self._pcm
        return data.tobytes()

    def set_volume(self, value):
        self._volume = min(1.0, max(0.0, float(value)))

    def get_volume(self):
        return self._volume

    def get_array(self):
        return (self._pcm[:, 0] if _config[2] == 1 else self._pcm).copy()

    def __array__(self, dtype=None, copy=None):
        array = self.get_array()
        return array.astype(dtype, copy=False) if dtype is not None else array


class _Channel:
    def __init__(self, channel_id):
        self.id = int(channel_id)
        self._sound = None
        self._queue = None
        self._position = 0
        self._loops = 0
        self._paused = False
        self._left = 1.0
        self._right = 1.0
        self._maxtime_frames = 0
        self._played_frames = 0
        self._endevent = 0

    def play(self, sound, loops=0, maxtime=0, fade_ms=0):
        if not isinstance(sound, Sound):
            raise TypeError("argument 1 must be Sound")
        self._sound = sound
        self._queue = None
        self._position = 0
        self._loops = int(loops)
        self._paused = False
        self._played_frames = 0
        self._maxtime_frames = max(0, int(float(maxtime) * _config[0] / 1000))

    def stop(self):
        self._sound = None
        self._queue = None
        self._position = 0
        self._played_frames = 0

    def pause(self):
        self._paused = True

    def unpause(self):
        self._paused = False

    def fadeout(self, time):
        self.stop()

    def set_volume(self, value, right=None):
        self._left = min(1.0, max(0.0, float(value)))
        self._right = self._left if right is None else min(1.0, max(0.0, float(right)))

    def get_volume(self):
        return self._left

    def get_busy(self):
        return self._sound is not None and not self._paused

    def get_sound(self):
        return self._sound

    def queue(self, sound):
        if not isinstance(sound, Sound):
            raise TypeError("queue argument must be a Sound")
        if self._sound is None:
            self.play(sound)
        else:
            self._queue = sound

    def get_queue(self):
        return self._queue

    def set_endevent(self, event_id=0):
        self._endevent = int(event_id)

    def get_endevent(self):
        return self._endevent

    def _advance_boundary(self):
        if self._sound is None:
            return
        if self._maxtime_frames and self._played_frames >= self._maxtime_frames:
            self.stop()
        elif self._position >= len(self._sound._pcm):
            if self._loops == -1 or self._loops > 0:
                if self._loops > 0:
                    self._loops -= 1
                self._position = 0
            elif self._queue is not None:
                self._sound, self._queue = self._queue, None
                self._position = 0
                self._played_frames = 0
                self._maxtime_frames = 0
            else:
                self.stop()


def Channel(channel_id):
    _require_init()
    channel_id = int(channel_id)
    if channel_id < 0 or channel_id >= _channel_count:
        raise IndexError("invalid channel index")
    return _channels[channel_id]


def set_num_channels(count):
    global _channel_count
    count = int(count)
    if count < 0:
        raise ValueError("negative channel count")
    for channel_id in tuple(_channels):
        if channel_id >= count:
            _channels[channel_id].stop()
            del _channels[channel_id]
    for channel_id in range(count):
        _channels.setdefault(channel_id, _Channel(channel_id))
    _channel_count = count


def get_num_channels():
    return _channel_count


def set_reserved(count):
    return min(max(0, int(count)), _channel_count)


def find_channel(force=False):
    _require_init()
    for channel in _channels.values():
        if channel._sound is None:
            return channel
    if force and _channels:
        return _channels[0]
    return None


def stop():
    for channel in _channels.values():
        channel.stop()


def pause():
    for channel in _channels.values():
        channel.pause()


def unpause():
    for channel in _channels.values():
        channel.unpause()


def fadeout(time):
    stop()


def get_busy():
    return any(channel.get_busy() for channel in _channels.values())


def _mix_pcm(arrays, positions, gains, frames):
    channels = _config[2]
    result = np.zeros((frames, channels), dtype=np.int16)
    if frames == 0 or not arrays:
        return result
    arrays = [np.ascontiguousarray(array, dtype=np.int16) for array in arrays]
    pointers = np.ascontiguousarray([_lib.addr(array) for array in arrays], dtype=np.int64)
    lengths = np.ascontiguousarray([len(array) for array in arrays], dtype=np.int64)
    positions = np.ascontiguousarray(positions, dtype=np.int64)
    gains = np.ascontiguousarray(gains, dtype=np.float64).reshape(len(arrays), channels)
    if not np.all(np.isfinite(gains)):
        raise ValueError("gains must be finite")
    _lib.lib().mpg_mix_i16(
        _lib.addr(pointers),
        _lib.addr(lengths),
        _lib.addr(positions),
        _lib.addr(gains),
        len(arrays),
        _lib.addr(result),
        frames,
        channels,
    )
    return result


def mix(sounds, volumes=None):
    """Mix Sound objects or int16 arrays and return saturated int16 PCM."""
    _require_init()
    sounds = list(sounds)
    arrays = [sound._pcm if isinstance(sound, Sound) else _coerce_pcm(sound) for sound in sounds]
    frames = max((len(array) for array in arrays), default=0)
    if volumes is None:
        volumes = [sound._volume if isinstance(sound, Sound) else 1.0 for sound in sounds]
    volumes = list(volumes)
    if len(volumes) != len(arrays):
        raise ValueError("volumes must have one entry per sound")
    gains = []
    for volume in volumes:
        if np.isscalar(volume):
            gains.append([float(volume)] * _config[2])
        else:
            pair = list(volume)
            if len(pair) != _config[2]:
                raise ValueError("per-channel volume does not match mixer channels")
            gains.append(pair)
    result = _mix_pcm(arrays, [0] * len(arrays), gains, frames)
    return result[:, 0] if _config[2] == 1 else result


def render(frames):
    """Render and advance the virtual channels by ``frames`` PCM frames."""
    _require_init()
    frames = int(frames)
    if frames < 0:
        raise ValueError("frames must be non-negative")
    result = np.zeros((frames, _config[2]), dtype=np.int16)
    cursor = 0
    while cursor < frames:
        active = [channel for channel in _channels.values() if channel.get_busy()]
        if not active:
            break
        remaining = []
        for channel in active:
            sound_remaining = len(channel._sound._pcm) - channel._position
            if channel._maxtime_frames:
                sound_remaining = min(
                    sound_remaining, channel._maxtime_frames - channel._played_frames
                )
            remaining.append(sound_remaining)
        block = min(frames - cursor, min(remaining))
        if block <= 0:
            for channel in active:
                channel._advance_boundary()
            continue
        gains = []
        for channel in active:
            sound_volume = channel._sound._volume
            gains.append(
                [channel._left * sound_volume]
                if _config[2] == 1
                else [channel._left * sound_volume, channel._right * sound_volume]
            )
        result[cursor : cursor + block] = _mix_pcm(
            [channel._sound._pcm for channel in active],
            [channel._position for channel in active],
            gains,
            block,
        )
        for channel in active:
            channel._position += block
            channel._played_frames += block
            channel._advance_boundary()
        cursor += block
    return result[:, 0] if _config[2] == 1 else result


def resample(sound_or_array, source_rate, target_rate):
    """Linearly resample signed int16 PCM; return a Sound only for Sound input."""
    _require_init()
    is_sound = isinstance(sound_or_array, Sound)
    source = sound_or_array._pcm if is_sound else _coerce_pcm(sound_or_array)
    source_rate, target_rate = int(source_rate), int(target_rate)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    frames = int(round(len(source) * target_rate / source_rate))
    result = np.empty((frames, _config[2]), dtype=np.int16)
    if frames and len(source):
        _lib.lib().mpg_resample_linear_i16(
            _lib.addr(source),
            len(source),
            _lib.addr(result),
            frames,
            _config[2],
            source_rate / target_rate,
        )
    if is_sound:
        return Sound(array=result[:, 0] if _config[2] == 1 else result)
    return result[:, 0] if _config[2] == 1 else result
