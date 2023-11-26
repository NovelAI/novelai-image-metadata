# Requirements: pip install pynacl pydantic
from PIL import Image
import argparse
import numpy as np
import json
import base64
from nacl.encoding import Base64Encoder
from nacl.signing import VerifyKey
from typing import Union
import os
import sys
from typing import Any
from PIL import Image
import numpy as np
import gzip
import json
import yaml
from typing import Union

class LSBExtractor:
    def __init__(self, data):
        self.data = data
        self.rows, self.cols, self.dim = data.shape
        self.bits = 0
        self.byte = 0
        self.row = 0
        self.col = 0

    def _extract_next_bit(self):
        if self.row < self.rows and self.col < self.cols:
            bit = self.data[self.row, self.col, self.dim - 1] & 1
            self.bits += 1
            self.byte <<= 1
            self.byte |= bit
            self.row += 1
            if self.row == self.rows:
                self.row = 0
                self.col += 1

    def get_one_byte(self):
        while self.bits < 8:
            self._extract_next_bit()
        byte = bytearray([self.byte])
        self.bits = 0
        self.byte = 0
        return byte

    def get_next_n_bytes(self, n):
        bytes_list = bytearray()
        for _ in range(n):
            byte = self.get_one_byte()
            if not byte:
                break
            bytes_list.extend(byte)
        return bytes_list

    def read_32bit_integer(self):
        bytes_list = self.get_next_n_bytes(4)
        if len(bytes_list) == 4:
            integer_value = int.from_bytes(bytes_list, byteorder='big')
            return integer_value
        else:
            return None

def extract_image_metadata(image: Union[Image.Image, np.ndarray]) -> dict:
    if isinstance(image, Image.Image):
        image = np.array(image)

    assert image.shape[-1] == 4 and len(image.shape) == 3, "image format"
    reader = LSBExtractor(image)
    magic = "stealth_pngcomp"
    read_magic = reader.get_next_n_bytes(len(magic)).decode("utf-8")
    assert magic == read_magic, "magic number"
    read_len = reader.read_32bit_integer() // 8
    json_data = reader.get_next_n_bytes(read_len)
    json_data = json.loads(gzip.decompress(json_data).decode("utf-8"))
    if "Comment" in json_data:
        json_data["Comment"] = json.loads(json_data["Comment"])

    return json_data

def verify_image_is_novelai(image: Union[Image.Image, np.ndarray], verify_key_hex: str="Y2JcQAOhLwzwSDUJPNgL04nS0Tbqm7cSRc4xk0vRMic=") -> bool:
    if isinstance(image, Image.Image):
        image = np.array(image)

    metadata = extract_image_metadata(image)

    if metadata is None:
        raise RuntimeError("No metadata found in image")

    if "Comment" not in metadata:
        raise RuntimeError("Comment not in metadata")

    comment = metadata["Comment"]
    if "signed_hash" not in comment:
        raise RuntimeError("signed_hash not in comment")

    signed_hash = comment["signed_hash"].encode("utf-8")
    signed_hash = base64.b64decode(signed_hash)
    del comment["signed_hash"]

    verify_key_hex = verify_key_hex.encode("utf-8")
    verify_key = VerifyKey(verify_key_hex, encoder=Base64Encoder)
    image_and_comment = image[:, :, :3].tobytes() + json.dumps(comment).encode("utf-8")
    try:
        verify_key.verify(image_and_comment, signed_hash)
    except:
        return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to image to check")
    args = parser.parse_args()

    image_path = args.image_path
    print(f"Checking image at {image_path}")
    image = Image.open(image_path)
    image = np.array(image)
    is_novelai = verify_image_is_novelai(image)
    print(f"Is image NovelAI? {is_novelai}")
