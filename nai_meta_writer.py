import json
import gzip
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from nai_bch import fec_encode

def unbyteize(data):
    data = np.unpackbits(data, axis=1)
    data = data.reshape((-1, 8))
    data = data.T.reshape((-1,))
    return data

class LSBInjector:
    def __init__(self, data):
        self.data = data
        self.buffer = bytearray()

    def put_byte(self, byte):
        self.buffer.append(byte)

    def put_32bit_integer(self, integer_value):
        self.buffer.extend(integer_value.to_bytes(4, byteorder='big'))

    def put_bytes(self, bytes_list):
        self.buffer.extend(bytes_list)

    def put_string(self, string):
        self.put_bytes(string.encode('utf-8'))

    def finalize(self):
        buffer = np.frombuffer(self.buffer, dtype=np.uint8)
        buffer = np.unpackbits(buffer)
        data = self.data[..., -1].T
        h, w = data.shape
        data = data.reshape((-1,))
        data[:] = 0xff
        buf_len = buffer.shape[0]
        data[:buf_len] = 0xfe
        data[:buf_len] = np.bitwise_or(data[:buf_len], buffer)
        data = data.reshape((h, w)).T
        self.data[..., -1] = data

def serialize_metadata(metadata: PngInfo) -> bytes:
    # Extract metadata from PNG chunks
    data = {
        k: v
        for k, v in [
            data[1]
            .decode("latin-1" if data[0] == b"tEXt" else "utf-8")
            .split("\x00" if data[0] == b"tEXt" else "\x00\x00\x00\x00\x00")
            for data in metadata.chunks
            if data[0] == b"tEXt" or data[0] == b"iTXt"
        ]
    }
    # Save space by getting rid of reduntant metadata (Title is static)
    if "Title" in data:
        del data["Title"]
    # Encode and compress data using gzip
    data_encoded = json.dumps(data)
    return gzip.compress(bytes(data_encoded, "utf-8"))

def inject_data(image: Image.Image, data: PngInfo, raw_metadata: bytes = None) -> Image.Image:
    import time
    rgb = np.array(image.convert('RGB'))
    image = image.convert('RGBA')
    w, h = image.size
    pixels = np.array(image)
    injector = LSBInjector(pixels)
    injector.put_string("stealth_pngcomp")
    if raw_metadata is not None:
        data = raw_metadata
    else:
        data = serialize_metadata(data)
    injector.put_32bit_integer(len(data) * 8)
    injector.put_bytes(data)
    fec_data = fec_encode(bytearray(rgb.tobytes()), w, h)
    injector.put_32bit_integer(len(fec_data) * 8)
    injector.put_bytes(fec_data)
    injector.finalize()
    return Image.fromarray(injector.data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inject metadata into PNG image')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input image')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output image')
    parser.add_argument('-m', '--metadata', type=str, required=True, help='Metadata entries in the form: key=value|key2=value2|...')
    args = parser.parse_args()

    image = Image.open(args.input)
    metadata = PngInfo()
    for entry in args.metadata.split('|'):
        key, value = entry.split('=')
        metadata.add_text(key, value)
    image = inject_data(image, metadata)
    image.save(args.output, "PNG", pnginfo=metadata)
