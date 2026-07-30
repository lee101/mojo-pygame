"""C ABI for the Python bindings. Python owns every buffer."""

from kernels import (
    BPtr,
    F64Ptr,
    I16Ptr,
    I64Ptr,
    blit_rgba,
    group_collisions,
    mask_overlap,
    mix_i16,
    rect_collisions,
    resample_linear_i16,
)


@export("mpg_blit_rgba")
def mpg_blit_rgba(
    src: Int,
    dst: Int,
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
    src_has_alpha: Int,
    dst_has_alpha: Int,
    use_colorkey: Int,
    key_r: Int,
    key_g: Int,
    key_b: Int,
) abi("C"):
    blit_rgba(
        BPtr(unsafe_from_address=src),
        BPtr(unsafe_from_address=dst),
        src_stride,
        dst_stride,
        src_x,
        src_y,
        dst_x,
        dst_y,
        width,
        height,
        mode,
        surface_alpha,
        src_has_alpha != 0,
        dst_has_alpha != 0,
        use_colorkey != 0,
        key_r,
        key_g,
        key_b,
    )


@export("mpg_rect_collisions")
def mpg_rect_collisions(target: Int, rects: Int, count: Int, hits: Int) abi("C") -> Int:
    return rect_collisions(
        I64Ptr(unsafe_from_address=target),
        I64Ptr(unsafe_from_address=rects),
        count,
        BPtr(unsafe_from_address=hits),
    )


@export("mpg_group_collisions")
def mpg_group_collisions(a: Int, na: Int, b: Int, nb: Int, hits: Int) abi("C") -> Int:
    return group_collisions(
        I64Ptr(unsafe_from_address=a),
        na,
        I64Ptr(unsafe_from_address=b),
        nb,
        BPtr(unsafe_from_address=hits),
    )


@export("mpg_mask_overlap")
def mpg_mask_overlap(
    a: Int,
    aw: Int,
    ah: Int,
    b: Int,
    bw: Int,
    bh: Int,
    offset_x: Int,
    offset_y: Int,
    first: Int,
) abi("C") -> Int:
    return mask_overlap(
        BPtr(unsafe_from_address=a),
        aw,
        ah,
        BPtr(unsafe_from_address=b),
        bw,
        bh,
        offset_x,
        offset_y,
        I64Ptr(unsafe_from_address=first),
        BPtr(unsafe_from_address=a),
        False,
    )


@export("mpg_mask_overlap_into")
def mpg_mask_overlap_into(
    a: Int,
    aw: Int,
    ah: Int,
    b: Int,
    bw: Int,
    bh: Int,
    offset_x: Int,
    offset_y: Int,
    first: Int,
    overlap_dst: Int,
) abi("C") -> Int:
    return mask_overlap(
        BPtr(unsafe_from_address=a),
        aw,
        ah,
        BPtr(unsafe_from_address=b),
        bw,
        bh,
        offset_x,
        offset_y,
        I64Ptr(unsafe_from_address=first),
        BPtr(unsafe_from_address=overlap_dst),
        True,
    )


@export("mpg_mix_i16")
def mpg_mix_i16(
    inputs: Int,
    lengths: Int,
    positions: Int,
    gains: Int,
    stream_count: Int,
    dst: Int,
    frames: Int,
    channels: Int,
) abi("C"):
    mix_i16(
        I64Ptr(unsafe_from_address=inputs),
        I64Ptr(unsafe_from_address=lengths),
        I64Ptr(unsafe_from_address=positions),
        F64Ptr(unsafe_from_address=gains),
        stream_count,
        I16Ptr(unsafe_from_address=dst),
        frames,
        channels,
    )


@export("mpg_resample_linear_i16")
def mpg_resample_linear_i16(
    src: Int,
    src_frames: Int,
    dst: Int,
    dst_frames: Int,
    channels: Int,
    rate_ratio: Float64,
) abi("C"):
    resample_linear_i16(
        I16Ptr(unsafe_from_address=src),
        src_frames,
        I16Ptr(unsafe_from_address=dst),
        dst_frames,
        channels,
        rate_ratio,
    )
