import sys
from typing import Union
from PIL import Image
import numpy as np
import gzip
import json
import yaml

def byteize(alpha):
    alpha = alpha.T.reshape((-1,))
    alpha = alpha[:(alpha.shape[0] // 8) * 8]
    alpha = np.bitwise_and(alpha, 1)
    alpha = alpha.reshape((-1, 8))
    alpha = np.packbits(alpha, axis=1)
    return alpha

class LSBExtractor:
    def __init__(self, data):
        self.data = byteize(data[..., -1])
        self.pos = 0

    def get_one_byte(self):
        byte = self.data[self.pos]
        self.pos += 1
        return byte

    def get_next_n_bytes(self, n):
        n_bytes = self.data[self.pos:self.pos + n]
        self.pos += n
        return bytearray(n_bytes)

    def read_32bit_integer(self):
        bytes_list = self.get_next_n_bytes(4)
        if len(bytes_list) == 4:
            integer_value = int.from_bytes(bytes_list, byteorder='big')
            return integer_value
        else:
            return None

def extract_image_metadata(image: Union[Image.Image, np.ndarray], get_fec: bool = False) -> dict:
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGBA"))

    assert image.shape[-1] == 4 and len(image.shape) == 3, "image format"
    reader = LSBExtractor(image)
    magic = "stealth_pngcomp"
    read_magic = reader.get_next_n_bytes(len(magic)).decode("utf-8")
    assert magic == read_magic, "magic number"
    read_len = reader.read_32bit_integer() // 8
    json_data = reader.get_next_n_bytes(read_len)
    json_data = json.loads(gzip.decompress(json_data).decode("utf-8"))
    if "Comment" in json_data and isinstance(json_data["Comment"], str):
        json_data["Comment"] = json.loads(json_data["Comment"])

    if not get_fec:
        return json_data

    fec_len = reader.read_32bit_integer()
    fec_data = None
    if fec_len != 0xffffffff:
        fec_data = reader.get_next_n_bytes(fec_len // 8)

    return json_data, fec_data

if __name__ == "__main__":
    for i, fn in enumerate(sys.argv[1:]):
        indent = False
        if len(sys.argv) > 2:
            if i > 0:
                print("\n")
            print(fn + ":")
            indent = True
        try:
            json_data = extract_image_metadata(Image.open(fn))
            yaml_data = yaml.dump(json_data, default_flow_style=False, sort_keys=False, width=float("inf"))
            if indent:
                yaml_data = '\n'.join([4 * ' ' + line for line in yaml.splitlines()])
            print(yaml_data)
        except Exception as e:
            print("failed: " + str(e))

