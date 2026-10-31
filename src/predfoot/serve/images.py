"""Checks on the images uploaded to ``/v1/calibrate``, made before any pixel is decoded."""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray

#: Image formats accepted by /v1/calibrate (Pillow's format name → media type). Pillow calls a JPEG that
#: carries a multi-picture (MPF) segment, as many phone and camera JPEGs do, "MPO".
MEDIA_TYPES = {"JPEG": "image/jpeg", "MPO": "image/jpeg", "PNG": "image/png"}
#: The only Pillow parsers that ever run on uploads.
PARSERS = ("JPEG", "PNG")
#: Uploads above this size are refused (20 MiB).
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
#: Images with more pixels are refused before decoding: a small PNG can decode to gigabytes.
MAX_PIXELS = 4096 * 4096


class ImageRejected(ValueError):
    """An upload that is not an acceptable image; ``status_code`` is the HTTP status to answer with."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def inspect_image(
    data: bytes, declared_type: str | None, max_pixels: int = MAX_PIXELS
) -> tuple[str, tuple[int, int]]:
    """Format and (width, height) of an upload, read from its header only.

    Raises :class:`ImageRejected` with 415 when the declared media type is not image/jpeg or image/png, when
    the bytes are not a JPEG or PNG image or when they do not match the declared type, and with 413 above
    ``max_pixels``.
    """
    from PIL import Image, UnidentifiedImageError

    media = (declared_type or "").split(";")[0].strip().lower()
    if media not in MEDIA_TYPES.values():
        raise ImageRejected(415, f"unsupported media type {media or 'none'!r}: send image/jpeg or image/png")
    try:
        with Image.open(io.BytesIO(data), formats=PARSERS) as img:
            fmt, (width, height) = img.format, img.size
    except Image.DecompressionBombError as err:
        raise ImageRejected(413, f"image too large: {err}") from err
    except (UnidentifiedImageError, OSError, ValueError) as err:
        raise ImageRejected(415, "the upload is not a readable JPEG or PNG image") from err
    if fmt not in MEDIA_TYPES:
        raise ImageRejected(415, f"unsupported image format {fmt}: send JPEG or PNG")
    if MEDIA_TYPES[fmt] != media:
        raise ImageRejected(415, f"the upload was declared {media} but holds a {fmt} image")
    if width * height > max_pixels:
        raise ImageRejected(413, f"image too large: {width} × {height} pixels, at most {max_pixels}")
    return fmt, (width, height)


def decode_rgb(data: bytes, size: tuple[int, int] | None = None) -> NDArray[np.uint8]:
    """Pixels (H, W, 3) uint8 RGB, decoded with OpenCV as in the offline pipeline (keypoints.cache).

    The pixels keep the grid they are stored in: an EXIF orientation tag is ignored, so pixel coordinates
    refer to the stored (width, height) that the header gives, whether or not the client's own decoder
    applies the tag. (The SN-GSR frames of the offline cache carry no such tag.) With ``size``, raises
    :class:`ImageRejected` when the decoded image does not have that (width, height).
    """
    import cv2

    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    if bgr is None:
        raise ImageRejected(415, "the image could not be decoded")
    if size is not None and (bgr.shape[1], bgr.shape[0]) != tuple(size):
        got = f"{bgr.shape[1]} × {bgr.shape[0]}"
        raise ImageRejected(415, f"the image decodes to {got} pixels, its header says {size[0]} × {size[1]}")
    return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
