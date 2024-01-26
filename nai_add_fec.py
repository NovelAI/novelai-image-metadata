import json
import gzip
import io

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from nai_meta_writer import inject_data
from nai_meta import extract_image_metadata

def add_fec(image: Image.Image, replace_rgb=None) -> Image.Image:
    metadata = PngInfo()
    for k, v in image.info.items():
        metadata.add_text(k, v)
    alpha_metadata = gzip.compress(json.dumps(extract_image_metadata(image)).encode())
    if replace_rgb is not None:
        rgb = np.array(image)
        rgb[:, :, :3] = replace_rgb
        image = Image.fromarray(rgb)
    data = inject_data(image, None, raw_metadata=alpha_metadata)
    png = io.BytesIO()
    data.save(png, format="PNG", pnginfo=metadata)
    return png.getvalue()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inject forward error correction data into PNG image')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input image')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output image')
    args = parser.parse_args()

    image = Image.open(args.input)
    image = add_fec(image)
    with open(args.output, "wb") as f:
        f.write(image)