"""Pixel, collision, and PCM kernels used by the C ABI."""

from std.algorithm import parallelize
from std.gpu.host import DeviceContext
from std.sys import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime I16Ptr = UnsafePointer[Int16, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]

comptime MIX_PARALLEL_WORK = 16777216
comptime MIX_TASK_SAMPLES = 16384


def clamp_u8(value: Int) -> UInt8:
    if value < 0:
        return UInt8(0)
    if value > 255:
        return UInt8(255)
    return UInt8(value)


def legacy_alpha(source: Int, dest: Int, alpha: Int) -> UInt8:
    if alpha <= 0:
        return UInt8(dest)
    if alpha >= 255:
        return UInt8(source)
    return UInt8(((dest << 8) + (source - dest) * alpha + source) >> 8)


def sdl2_alpha(source: Int, dest: Int, alpha: Int) -> UInt8:
    if alpha <= 0:
        return UInt8(dest)
    if alpha >= 255:
        return UInt8(source)
    return UInt8((source * alpha + dest * (255 - alpha)) >> 8)


def combined_alpha(pixel_alpha: Int, surface_alpha: Int) -> Int:
    if surface_alpha >= 255:
        return pixel_alpha
    return (pixel_alpha * (surface_alpha + 1)) >> 8


def blend_channel(source: Int, dest: Int, mode: Int) -> UInt8:
    if mode == 1 or mode == 6:
        return clamp_u8(source + dest)
    if mode == 2 or mode == 7:
        return clamp_u8(dest - source)
    if mode == 3 or mode == 8:
        return UInt8((source * dest + 255) >> 8)
    if mode == 4 or mode == 9:
        return UInt8(source if source < dest else dest)
    return UInt8(source if source > dest else dest)


def blit_row(
    src: BPtr,
    dst: BPtr,
    src_stride: Int,
    dst_stride: Int,
    src_x: Int,
    src_y: Int,
    dst_x: Int,
    dst_y: Int,
    width: Int,
    y: Int,
    mode: Int,
    surface_alpha: Int,
    src_has_alpha: Bool,
    dst_has_alpha: Bool,
    use_colorkey: Bool,
    key_r: Int,
    key_g: Int,
    key_b: Int,
):
    var si = (src_y + y) * src_stride + src_x * 4
    var di = (dst_y + y) * dst_stride + dst_x * 4
    if mode == 6 and not use_colorkey:
        comptime W = simd_width_of[DType.uint8]()
        var byte_count = width * 4
        var byte = 0
        while byte + W <= byte_count:
            var source = src.load[width=W](si + byte).cast[DType.uint16]()
            var dest = dst.load[width=W](di + byte).cast[DType.uint16]()
            var blended = min(
                source + dest, SIMD[DType.uint16, W](UInt16(255))
            ).cast[DType.uint8]()
            dst.store(di + byte, blended)
            byte += W
        while byte < byte_count:
            dst[di + byte] = clamp_u8(
                Int(src[si + byte]) + Int(dst[di + byte])
            )
            byte += 1
        return
    var x = 0
    if (
        mode == 0
        and surface_alpha >= 0
        and not use_colorkey
        and src_has_alpha
        and dst_has_alpha
    ):
        comptime W = simd_width_of[DType.float64]()
        comptime N = W * 4
        var alpha_lanes = SIMD[DType.bool, N](fill=False)
        comptime for lane in range(N):
            if lane % 4 == 3:
                alpha_lanes[lane] = True
        while x + W <= width:
            var source = src.load[width=N](si).cast[DType.int32]()
            var dest = dst.load[width=N](di).cast[DType.int32]()
            var alpha = SIMD[DType.int32, N]()
            var dest_alpha = SIMD[DType.int32, N]()
            comptime for lane in range(N):
                alpha[lane] = source[(lane // 4) * 4 + 3]
                dest_alpha[lane] = dest[(lane // 4) * 4 + 3]
            if surface_alpha < 255:
                alpha = (alpha * Int32(surface_alpha + 1)) >> Int32(8)
            var legacy = (
                (dest << Int32(8)) + (source - dest) * alpha + source
            ) >> Int32(8)
            var rgb = alpha.eq(Int32(0)).select(
                dest, alpha.ge(Int32(255)).select(source, legacy)
            )
            var transparent_dest = dest_alpha.eq(Int32(0)) & ~alpha_lanes
            rgb = transparent_dest.select(source, rgb)
            var alpha_result = alpha + dest - (
                (alpha * dest) // SIMD[DType.int32, N](Int32(255))
            )
            var blended = alpha_lanes.select(alpha_result, rgb).cast[DType.uint8]()
            dst.store(di, blended)
            x += W
            si += N
            di += N
    while x < width:
        var sr = Int(src[si])
        var sg = Int(src[si + 1])
        var sb = Int(src[si + 2])
        if use_colorkey and sr == key_r and sg == key_g and sb == key_b:
            si += 4
            di += 4
            x += 1
            continue
        if mode == 0 and surface_alpha < 0:
            dst[di] = src[si]
            dst[di + 1] = src[si + 1]
            dst[di + 2] = src[si + 2]
            dst[di + 3] = src[si + 3] if dst_has_alpha else UInt8(255)
            si += 4
            di += 4
            x += 1
            continue
        var sa = Int(src[si + 3]) if src_has_alpha else 255
        if mode == 0 or mode == 18:
            sa = combined_alpha(sa, surface_alpha)
            var dr = Int(dst[di])
            var dg = Int(dst[di + 1])
            var db = Int(dst[di + 2])
            var da = Int(dst[di + 3])
            if mode == 0 and not use_colorkey and dst_has_alpha and da == 0:
                dst[di] = UInt8(sr)
                dst[di + 1] = UInt8(sg)
                dst[di + 2] = UInt8(sb)
            elif mode == 18:
                dst[di] = sdl2_alpha(sr, dr, sa)
                dst[di + 1] = sdl2_alpha(sg, dg, sa)
                dst[di + 2] = sdl2_alpha(sb, db, sa)
            else:
                dst[di] = legacy_alpha(sr, dr, sa)
                dst[di + 1] = legacy_alpha(sg, dg, sa)
                dst[di + 2] = legacy_alpha(sb, db, sa)
            if dst_has_alpha:
                if mode == 18:
                    dst[di + 3] = UInt8((255 * sa + da * (255 - sa)) >> 8)
                else:
                    dst[di + 3] = clamp_u8(sa + da - ((sa * da) // 255))
            else:
                dst[di + 3] = UInt8(255)
        elif mode == 17:
            sa = combined_alpha(sa, surface_alpha)
            if sa == 0:
                pass
            elif sa >= 255:
                dst[di] = src[si]
                dst[di + 1] = src[si + 1]
                dst[di + 2] = src[si + 2]
            else:
                for c in range(3):
                    var s = Int(src[si + c])
                    var d = Int(dst[di + c])
                    dst[di + c] = clamp_u8(s + ((d * (255 - sa) + 255) >> 8))
            if dst_has_alpha and sa != 0:
                var da = Int(dst[di + 3])
                dst[di + 3] = clamp_u8(sa + da - ((sa * da + 255) >> 8))
            else:
                if not dst_has_alpha:
                    dst[di + 3] = UInt8(255)
        else:
            dst[di] = blend_channel(sr, Int(dst[di]), mode)
            dst[di + 1] = blend_channel(sg, Int(dst[di + 1]), mode)
            dst[di + 2] = blend_channel(sb, Int(dst[di + 2]), mode)
            if mode >= 6:
                dst[di + 3] = blend_channel(Int(src[si + 3]), Int(dst[di + 3]), mode)
        si += 4
        di += 4
        x += 1


def blit_rgba(
    src: BPtr,
    dst: BPtr,
    src_stride: Int,
    dst_stride: Int,
    src_x: Int,
    src_y: Int,
    dst_x: Int,
    dst_y: Int,
    width: Int,
    height: Int,
    mode: Int,
    surface_alpha: Int,
    src_has_alpha: Bool,
    dst_has_alpha: Bool,
    use_colorkey: Bool,
    key_r: Int,
    key_g: Int,
    key_b: Int,
):
    @parameter
    def row_work(y: Int):
        blit_row(
            src,
            dst,
            src_stride,
            dst_stride,
            src_x,
            src_y,
            dst_x,
            dst_y,
            width,
            y,
            mode,
            surface_alpha,
            src_has_alpha,
            dst_has_alpha,
            use_colorkey,
            key_r,
            key_g,
            key_b,
        )

    # Blits modify an existing destination. Retrying after a partially failed
    # parallel launch would apply non-idempotent blending twice, so keep this
    # operation synchronous across the C ABI.
    for y in range(height):
        row_work(y)


def rect_collisions(target: I64Ptr, rects: I64Ptr, count: Int, hits: BPtr) -> Int:
    var tx = target[0]
    var ty = target[1]
    var tr = tx + target[2]
    var tb = ty + target[3]
    var total = 0
    for i in range(count):
        var j = i * 4
        var x = rects[j]
        var y = rects[j + 1]
        var hit = target[2] > 0 and target[3] > 0 and rects[j + 2] > 0 and rects[j + 3] > 0 and x < tr and x + rects[j + 2] > tx and y < tb and y + rects[j + 3] > ty
        hits[i] = UInt8(1) if hit else UInt8(0)
        if hit:
            total += 1
    return total


def group_collisions(a: I64Ptr, na: Int, b: I64Ptr, nb: Int, hits: BPtr) -> Int:
    var total = 0
    for i in range(na):
        var ai = i * 4
        var ax = a[ai]
        var ay = a[ai + 1]
        var ar = ax + a[ai + 2]
        var ab = ay + a[ai + 3]
        for j in range(nb):
            var bi = j * 4
            var bx = b[bi]
            var by = b[bi + 1]
            var hit = a[ai + 2] > 0 and a[ai + 3] > 0 and b[bi + 2] > 0 and b[bi + 3] > 0 and bx < ar and bx + b[bi + 2] > ax and by < ab and by + b[bi + 3] > ay
            hits[i * nb + j] = UInt8(1) if hit else UInt8(0)
            if hit:
                total += 1
    return total


def mask_overlap(
    a: BPtr,
    aw: Int,
    ah: Int,
    b: BPtr,
    bw: Int,
    bh: Int,
    offset_x: Int,
    offset_y: Int,
    first: I64Ptr,
    overlap_dst: BPtr,
    write_overlap: Bool,
) -> Int:
    var x0 = offset_x if offset_x > 0 else 0
    var y0 = offset_y if offset_y > 0 else 0
    var x1 = offset_x + bw if offset_x + bw < aw else aw
    var y1 = offset_y + bh if offset_y + bh < ah else ah
    first[0] = -1
    first[1] = -1
    if x1 <= x0 or y1 <= y0:
        return 0
    var total = 0
    comptime W = simd_width_of[DType.uint8]()
    for y in range(y0, y1):
        var ai = y * aw + x0
        var bi = (y - offset_y) * bw + x0 - offset_x
        var x = x0
        while x + W <= x1:
            var both = a.load[width=W](ai).ne(UInt8(0)) & (
                b.load[width=W](bi).ne(UInt8(0))
            )
            var values = both.select(
                SIMD[DType.uint8, W](UInt8(1)),
                SIMD[DType.uint8, W](UInt8(0)),
            )
            var block_total = Int(values.reduce_add())
            if total == 0 and block_total != 0:
                for lane in range(W):
                    if both[lane]:
                        first[0] = Int64(x + lane)
                        first[1] = Int64(y)
                        break
            if write_overlap:
                overlap_dst.store(ai, values)
            total += block_total
            x += W
            ai += W
            bi += W
        while x < x1:
            if a[ai] != 0 and b[bi] != 0:
                if total == 0:
                    first[0] = Int64(x)
                    first[1] = Int64(y)
                if write_overlap:
                    overlap_dst[ai] = UInt8(1)
                total += 1
            x += 1
            ai += 1
            bi += 1
    return total


def mix_i16_range(
    inputs: I64Ptr,
    lengths: I64Ptr,
    positions: I64Ptr,
    gains: F64Ptr,
    stream_count: Int,
    dst: I16Ptr,
    channels: Int,
    start_sample: Int,
    end_sample: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var sample_index = start_sample
    while sample_index + W <= end_sample:
        var mixed = SIMD[DType.float64, W](Float64(0.0))
        for stream in range(stream_count):
            var source_limit = Int(lengths[stream]) * channels
            var source_index = Int(positions[stream]) * channels + sample_index
            if source_index >= source_limit:
                continue
            var source = I16Ptr(unsafe_from_address=Int(inputs[stream]))
            var samples = SIMD[DType.float64, W](Float64(0.0))
            if source_index + W <= source_limit:
                samples = source.load[width=W](source_index).cast[DType.float64]()
            else:
                for lane in range(W):
                    if source_index + lane < source_limit:
                        samples[lane] = Float64(source[source_index + lane])
            if channels == 1 or (
                channels == 2
                and gains[stream * channels] == gains[stream * channels + 1]
            ):
                mixed += samples * gains[stream * channels]
            else:
                var gain_values = SIMD[DType.float64, W]()
                for lane in range(W):
                    gain_values[lane] = gains[
                        stream * channels + ((sample_index + lane) % channels)
                    ]
                mixed += samples * gain_values
        var clipped = mixed.clamp(
            SIMD[DType.float64, W](Float64(-32768.0)),
            SIMD[DType.float64, W](Float64(32767.0)),
        )
        dst.store(sample_index, clipped.cast[DType.int16]())
        sample_index += W
    while sample_index < end_sample:
        var frame = sample_index // channels
        var channel = sample_index % channels
        var sample = 0.0
        for stream in range(stream_count):
            var source_frame = Int(positions[stream]) + frame
            if source_frame < Int(lengths[stream]):
                var source = I16Ptr(unsafe_from_address=Int(inputs[stream]))
                sample += Float64(source[source_frame * channels + channel]) * gains[
                    stream * channels + channel
                ]
        var value = Int(sample)
        if value > 32767:
            value = 32767
        elif value < -32768:
            value = -32768
        dst[sample_index] = Int16(value)
        sample_index += 1


def mix_i16(
    inputs: I64Ptr,
    lengths: I64Ptr,
    positions: I64Ptr,
    gains: F64Ptr,
    stream_count: Int,
    dst: I16Ptr,
    frames: Int,
    channels: Int,
):
    var sample_count = frames * channels

    @parameter
    def work(task: Int):
        var start = task * MIX_TASK_SAMPLES
        var end = min(start + MIX_TASK_SAMPLES, sample_count)
        mix_i16_range(
            inputs,
            lengths,
            positions,
            gains,
            stream_count,
            dst,
            channels,
            start,
            end,
        )

    if sample_count * stream_count >= MIX_PARALLEL_WORK:
        var task_count = (sample_count + MIX_TASK_SAMPLES - 1) // MIX_TASK_SAMPLES
        try:
            with DeviceContext(api="cpu") as ctx:
                _ = ctx
                parallelize[work](task_count)
        except:
            mix_i16_range(
                inputs,
                lengths,
                positions,
                gains,
                stream_count,
                dst,
                channels,
                0,
                sample_count,
            )
    else:
        mix_i16_range(
            inputs,
            lengths,
            positions,
            gains,
            stream_count,
            dst,
            channels,
            0,
            sample_count,
        )


def resample_linear_i16(
    src: I16Ptr,
    src_frames: Int,
    dst: I16Ptr,
    dst_frames: Int,
    channels: Int,
    rate_ratio: Float64,
):
    if src_frames == 0:
        return
    for frame in range(dst_frames):
        var position = Float64(frame) * rate_ratio
        var left = Int(position)
        if left >= src_frames - 1:
            left = src_frames - 1
            for channel in range(channels):
                dst[frame * channels + channel] = src[left * channels + channel]
            continue
        var fraction = position - Float64(left)
        for channel in range(channels):
            var a = Float64(src[left * channels + channel])
            var b = Float64(src[(left + 1) * channels + channel])
            var value = Int(a + (b - a) * fraction)
            if value > 32767:
                value = 32767
            elif value < -32768:
                value = -32768
            dst[frame * channels + channel] = Int16(value)
