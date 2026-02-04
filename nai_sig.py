# Requirements: pip install pynacl pydantic numpy Pillow
import logging
import hashlib
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from json import loads as json_loads
from json import dumps as json_dumps
import base64
from nacl.encoding import Base64Encoder
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError
from typing import Optional, Any, Dict
from gzip import decompress as gzip_decompress
from dataclasses import dataclass, asdict
import io
import sys
import os
import argparse
from nai_meta import LSBExtractor

ImageArray = NDArray[np.int8]
log = logging.getLogger(__name__)


@dataclass
class Metadata:
    raw_bytes: bytes
    description: Optional[str]
    software: Optional[str]
    source: Optional[str]
    generation_time: Optional[str]
    # only Comment is required to be here for verification purposes
    # all other fields can be missing
    comment: str

    @classmethod
    def from_bytes(cls, raw_bytes: bytes) -> "Metadata":
        raw_dict = json_loads(raw_bytes)

        return cls(
            raw_bytes,
            description=raw_dict.get("Description"),
            software=raw_dict.get("Software"),
            source=raw_dict.get("Source"),
            generation_time=raw_dict.get("Generation time"),
            comment=raw_dict["Comment"],
        )

    @property
    def as_dict(self) -> Any:
        return {
            "Description": self.description,
            "Software": self.software,
            "Source": self.source,
            "Generation time": self.generation_time,
            "Comment": self.comment,
        }

    @property
    def as_json(self) -> Any:
        return json_loads(self.raw_bytes.decode("utf-8"))

    @property
    def decoded_comment(self) -> Any:
        decoded = json_loads(self.comment)
        if "Comment" in decoded:
            return json_loads(decoded["Comment"])
        return decoded

    @property
    def as_base64(self) -> str:
        return base64.b64encode(self.raw_bytes).decode("utf-8")


def extract_alpha_metadata_raw(image_bytes: ImageArray) -> bytes:
    """Extract raw metadata bytes from image (before any JSON parsing)."""
    assert isinstance(image_bytes, np.ndarray)

    assert image_bytes.shape[-1] == 4 and len(image_bytes.shape) == 3, "image format"
    reader = LSBExtractor(image_bytes)
    magic = "stealth_pngcomp"
    read_magic = reader.get_next_n_bytes(len(magic)).decode("utf-8")
    assert magic == read_magic, "magic number"
    read_len_int32 = reader.read_32bit_integer()
    assert read_len_int32 is not None
    read_len = read_len_int32 // 8
    json_data = reader.get_next_n_bytes(read_len)
    return gzip_decompress(json_data)


def extract_alpha_metadata(image_bytes: ImageArray) -> Metadata:
    """Extract and parse metadata from image. Comment is parsed as an object."""
    raw_bytes = extract_alpha_metadata_raw(image_bytes)
    return Metadata.from_bytes(raw_bytes)


def _verify_signed_hash(
    image_array: ImageArray, verify_key_hex: str, comment: Any
) -> bool:
    assert isinstance(image_array, np.ndarray)
    signed_hash = comment["signed_hash"].encode("utf-8")
    signed_hash = base64.b64decode(signed_hash)
    del comment["signed_hash"]

    verify_key_bytes = verify_key_hex.encode("utf-8")
    verify_key = VerifyKey(verify_key_bytes, encoder=Base64Encoder)
    image_and_comment = image_array[:, :, :3].tobytes() + json_dumps(comment).encode(
        "utf-8"
    )

    log.info(
        "verifying a signed hash with data: %r",
        hashlib.sha256(image_and_comment).hexdigest(),
    )
    try:
        _ = verify_key.verify(image_and_comment, signed_hash)
    except BadSignatureError:
        return False

    return True


def verify_metadata_is_novelai(
    image: Image.Image,
    image_array: ImageArray,
    metadata: Metadata,
    verify_key_hex: str = "Y2JcQAOhLwzwSDUJPNgL04nS0Tbqm7cSRc4xk0vRMic=",
) -> bool:
    assert isinstance(image, Image.Image)
    assert isinstance(image_array, np.ndarray)
    comment = metadata.decoded_comment
    if "signed_hash" not in comment:
        raise RuntimeError(f"signed_hash not in comment: {comment}")

    return _verify_signed_hash(image_array, verify_key_hex, comment)


def is_image_allowed_format(image: Image.Image) -> bool:
    return image.format in ("PNG", "WEBP")


# --- PNG tEXt metadata extraction and verification ---


def extract_png_text_metadata(image: Image.Image) -> Optional[Metadata]:
    """Extract and parse metadata from PNG tEXt chunks. Comment is parsed as an object."""
    if "Comment" not in image.info:
        return None

    # Collect all PNG text chunks into a dict (Comment stays as string)
    metadata = {}
    for key in [
        "Title",
        "Description",
        "Software",
        "Source",
        "Generation time",
        "Comment",
    ]:
        if key in image.info:
            metadata[key] = image.info[key]

    # Return as JSON bytes (matching alpha channel format)
    fake_raw_bytes = json_dumps(metadata).encode("utf-8")
    return Metadata.from_bytes(fake_raw_bytes)


# --- WebP EXIF metadata extraction and verification ---
# EXIF field mapping (matches buildEXIFMetadata in post.go):
#   Software -> Source
#   ImageDescription -> Description
#   Copyright -> Copyright
#   DocumentName -> Title
#   Artist -> Artist
#   UserComment (in EXIF IFD) -> Comment

from PIL.ExifTags import Base as ExifBase, IFD as ExifIFD


def _decode_user_comment(data: bytes) -> Optional[str]:
    """Decode EXIF UserComment field which has an 8-byte encoding prefix."""
    if len(data) < 8:
        return None

    # Check encoding prefix (first 8 bytes)
    encoding_prefix = data[:8]
    content = data[8:]

    # EXIF in its glory has not specified a canonical encoding of the UserComment field.
    # NAI encodes as ASCII and so we require the ASCII encoding magic value.
    #
    # if something re-encodes our metadata in a different charset, we should not allow it.

    # ASCII encoding: "ASCII\x00\x00\x00"
    if encoding_prefix == b"ASCII\x00\x00\x00":
        # Strip null bytes from the end
        return content.rstrip(b"\x00").decode("ascii", errors="replace")
    return None


def extract_exif_metadata(image: Image.Image) -> Optional[Metadata]:
    """Extract and parse metadata from WebP EXIF. Comment is parsed as an object."""
    exif_data = image.getexif()
    if not exif_data:
        return None

    metadata = {}

    # Map EXIF fields to PNG tEXt equivalent names
    if ExifBase.DocumentName in exif_data:
        metadata["Title"] = exif_data[ExifBase.DocumentName]
    if ExifBase.ImageDescription in exif_data:
        metadata["Description"] = exif_data[ExifBase.ImageDescription]
    if ExifBase.Software in exif_data:
        metadata["Source"] = exif_data[ExifBase.Software]
    if ExifBase.Artist in exif_data:
        metadata["Artist"] = exif_data[ExifBase.Artist]
    if ExifBase.Copyright in exif_data:
        metadata["Copyright"] = exif_data[ExifBase.Copyright]

    # Get EXIF sub-IFD for UserComment
    exif_ifd = exif_data.get_ifd(ExifIFD.Exif)
    if exif_ifd and ExifBase.UserComment in exif_ifd:
        user_comment_raw = exif_ifd[ExifBase.UserComment]
        comment = None
        if isinstance(user_comment_raw, bytes):
            comment = _decode_user_comment(user_comment_raw)
        else:
            comment = str(user_comment_raw)

        if comment:
            metadata["Comment"] = comment

    if "Comment" not in metadata:
        return None

    # Return as JSON bytes (matching alpha channel format)
    fake_raw_bytes = json_dumps(metadata).encode("utf-8")
    return Metadata.from_bytes(fake_raw_bytes)


@dataclass
class VerifyResult:
    success: bool = False
    error: Optional[str] = None

    # NAI contains two forms of metadata in images:
    # - Alpha channel via stealth_pngcomp algorithm (PNG and WEBP support Alpha, so they have this)
    # - file-level metadata (png tEXt of webp EXIF)
    #
    # these fields represent the extraction and verification of alpha channel metadata
    is_novelai_alpha: bool = False
    metadata_alpha: Optional[Any] = None
    # for metadata_alpha_raw, this returns the actual bytes as decoded
    # by the stealth_pngcomp algorithm
    metadata_alpha_raw: Optional[str] = None

    # if the file metadata (png tEXt, webp EXIF) was verified successfully
    is_novelai_file: bool = False
    metadata_file: Optional[Any] = None
    # for metadata_file_raw, this represents a "normalized" view of the metadata
    # to align it with how the alpha metadata is structured. this is done because
    # things like tEXt/EXIF encode Title, Description, Software etc differently
    metadata_file_raw: Optional[str] = None


class InputError(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description="Verify NovelAI image metadata")
    parser.add_argument("-i", "--input", help="Path to image file")
    args = parser.parse_args()

    if args.input:
        # Read from file
        image = Image.open(args.input)
    elif not sys.stdin.isatty():
        # Read from stdin
        image_bytes = sys.stdin.buffer.read()
        image_buffer = io.BytesIO(image_bytes)
        image = Image.open(image_buffer)
    else:
        raise InputError(
            "No input provided. Use -i <image path> or pipe an image to stdin."
        )

    if not is_image_allowed_format(image):
        raise InputError(f"Invalid format, got {image.format}")

    image_array = np.array(image)
    result = VerifyResult()

    # extract alpha and verify it (both PNG and WEBP have alpha metadata)
    alpha_metadata: Metadata = extract_alpha_metadata(image_array)
    result.metadata_alpha_raw = alpha_metadata.as_base64
    result.metadata_alpha = alpha_metadata.as_dict
    result.is_novelai_alpha = verify_metadata_is_novelai(
        image, image_array, alpha_metadata
    )

    # --- File-level metadata (PNG tEXt or WebP EXIF) ---
    if image.format == "PNG":
        png_metadata = extract_png_text_metadata(image)
        if png_metadata:
            result.metadata_file_raw = png_metadata.as_base64
            result.metadata_file = png_metadata.as_dict
            result.is_novelai_file = verify_metadata_is_novelai(
                image, image_array, png_metadata
            )

    elif image.format == "WEBP":
        exif_metadata = extract_exif_metadata(image)
        if exif_metadata:
            result.metadata_file_raw = exif_metadata.as_base64
            result.metadata_file = exif_metadata.as_dict
            result.is_novelai_file = verify_metadata_is_novelai(
                image, image_array, exif_metadata
            )

    result.success = True
    if os.environ.get("JSON", "0") == "1":
        print(json_dumps(asdict(result)))
    else:
        # NOTE: more strict checkers should do `and` instead of `or`
        print(f"Is image NovelAI? {result.is_novelai_alpha or result.is_novelai_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
