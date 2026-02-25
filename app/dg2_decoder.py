"""
dg2_decoder.py
--------------
Extracts the raw JPEG / JPEG2000 face image from a DG2 TLV byte structure
as read from an e-Passport NFC chip (ICAO 9303 Part 10).

DG2 structure (simplified):
  75 xx       — DG2 tag + length
    7F61 xx   — Biometric Information Group
      02 xx   — Number of instances
      7F60 xx — Biometric Information Template (one per face)
        A1 xx — Biometric Header Template
          ...   (version, type, subtype, creation date, validity, creator)
        5F2E xx — Biometric Data Block   ← the actual image data lives here
          OR
        7F2E xx — same, alternate tag

Inside the Biometric Data Block (ISO 19794-5 CBEFF wrapping):
  Bytes 0-3:   "FAC\x00"   — Format identifier
  Bytes 4-7:   "010\x00"   — Version
  Bytes 8-11:  uint32 BE   — Record length
  Bytes 12-13: uint16 BE   — Number of faces
  Bytes 14-79: 66-byte Facial Record Header
  Bytes 80+:   Facial Record Data (image data starts with JPEG/JPEG2000 magic)

We parse the TLV tree to find tag 5F2E or 7F2E, then skip the CBEFF header
to find the raw JPEG/JPEG2000 bytes.
"""

import struct
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# TLV parser
# ---------------------------------------------------------------------------

def _read_length(data: bytes, offset: int) -> Tuple[int, int]:
    """
    Parse ASN.1/BER length at offset.
    Returns (length_value, bytes_consumed).
    """
    first = data[offset]
    if first < 0x80:
        return first, 1
    num_bytes = first & 0x7F
    length = int.from_bytes(data[offset + 1: offset + 1 + num_bytes], "big")
    return length, 1 + num_bytes


def _read_tag(data: bytes, offset: int) -> Tuple[int, int]:
    """
    Parse ASN.1/BER tag at offset (handles 1- and 2-byte tags).
    Returns (tag_value, bytes_consumed).
    """
    first = data[offset]
    if (first & 0x1F) != 0x1F:
        return first, 1
    # Two-byte tag
    second = data[offset + 1]
    return (first << 8) | second, 2


def _find_tag(data: bytes, target_tag: int) -> Optional[bytes]:
    """
    Walk the TLV tree recursively and return the VALUE of the first
    occurrence of target_tag, or None if not found.
    """
    offset = 0
    while offset < len(data):
        try:
            tag, tag_len = _read_tag(data, offset)
            offset += tag_len
            length, len_len = _read_length(data, offset)
            offset += len_len
        except (IndexError, struct.error):
            break

        value = data[offset: offset + length]

        if tag == target_tag:
            return value

        # Recurse into constructed tags (bit 5 of first byte set, or known composites)
        first_byte = (tag >> 8) if tag > 0xFF else tag
        if (first_byte & 0x20) or tag in (0x7F61, 0x7F60, 0xA1, 0x02):
            result = _find_tag(value, target_tag)
            if result is not None:
                return result

        offset += length

    return None


# ---------------------------------------------------------------------------
# CBEFF / ISO 19794-5 image extraction
# ---------------------------------------------------------------------------

_JPEG_MAGIC      = b"\xff\xd8\xff"
_JPEG2000_MAGIC  = b"\x00\x00\x00\x0c\x6a\x50\x20\x20"   # JP2 signature box
_JPEG2000_MAGIC2 = b"\xff\x4f\xff\x51"                     # JPEG 2000 codestream


def _extract_image_from_cbeff(data: bytes) -> Optional[bytes]:
    """
    Given the raw content of a 5F2E/7F2E Biometric Data Block,
    scan forward until we find a JPEG or JPEG2000 magic sequence.
    """
    for i in range(len(data) - 4):
        chunk = data[i:]
        if (chunk[:3] == _JPEG_MAGIC
                or chunk[:8] == _JPEG2000_MAGIC
                or chunk[:4] == _JPEG2000_MAGIC2):
            return data[i:]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_dg2(dg2_bytes: bytes) -> Optional[bytes]:
    """
    Extract the raw JPEG or JPEG2000 face image from a DG2 byte string.

    Args:
        dg2_bytes: Raw bytes of the DG2 data group as read from the NFC chip.
                   May be hex-encoded string or raw bytes.

    Returns:
        Raw image bytes (JPEG or JPEG2000) ready to be decoded by OpenCV,
        or None if extraction failed.
    """
    # Accept hex-encoded string (as React Native may send it)
    if isinstance(dg2_bytes, str):
        try:
            dg2_bytes = bytes.fromhex(dg2_bytes)
        except ValueError:
            import base64
            try:
                dg2_bytes = base64.b64decode(dg2_bytes)
            except Exception:
                return None

    # Try tag 5F2E first (standard), then 7F2E (alternate)
    biometric_data = _find_tag(dg2_bytes, 0x5F2E)
    if biometric_data is None:
        biometric_data = _find_tag(dg2_bytes, 0x7F2E)

    if biometric_data is None:
        # Last resort: scan the whole DG2 for JPEG magic
        return _extract_image_from_cbeff(dg2_bytes)

    return _extract_image_from_cbeff(biometric_data)


def decode_dg2_to_cv2(dg2_bytes: bytes):
    """
    Extract face image from DG2 and decode it into a BGR numpy array
    suitable for use with OpenCV / MTCNN.

    Returns:
        numpy.ndarray (BGR) or None if extraction/decoding failed.
    """
    import cv2
    import numpy as np

    image_bytes = decode_dg2(dg2_bytes)
    if image_bytes is None:
        return None

    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image  # None if OpenCV couldn't decode it