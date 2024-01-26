from PIL import Image
import argparse
import numpy as np
import json
import base64
from nacl.encoding import Base64Encoder
from nacl.signing import VerifyKey
from PIL import Image
import numpy as np
import base64
import json

from nai_bch import fec_decode
from nai_add_fec import add_fec
from nai_meta import extract_image_metadata

def verify_latents(image: Image.Image, signed_hash: bytes ,verify_key: VerifyKey) -> bool:
    image.load()
    sig = None
    latents = None
    try:
        for cid, data, after_idat in image.private_chunks:
            if after_idat:
                if cid == b'ltns':
                    sig = data
                elif cid == b'ltnt':
                    latents = data
    except:
        return True, False, None
    if sig is None or latents is None:
        return True, False, None
    if not sig.startswith(b'NovelAI_ltntsig'):
        return False, False, None
    sig = sig[len(b'NovelAI_ltntsig'):]
    if not latents.startswith(b'NovelAI_latents'):
        return False, False, None
    latents = latents[len(b'NovelAI_latents'):]
    if len(sig) != 64:
        return False, False, None
    w, h = image.size
    base_size = (w // 8) * (h // 8) * 4
    if len(latents) != base_size * 4 and len(latents) != base_size * 2:
        return False, False, None
    float_dim = 4 if len(latents) == base_size * 4 else 2
    try:
        verify_key.verify(signed_hash + latents, sig)
        return True, True, (float_dim, latents)
    except:
        return False, False, None

def verify_image_is_novelai(image: Image.Image, verify_key_hex: str="Y2JcQAOhLwzwSDUJPNgL04nS0Tbqm7cSRc4xk0vRMic=", output_fixed: bool=False) -> bool:
    metadata, fec_data = extract_image_metadata(image, get_fec=True)
    w, h = image.size

    if metadata is None:
        #raise RuntimeError("No metadata found in image")
        return False, False, None, None, None

    if "Comment" not in metadata:
        #raise RuntimeError("Comment not in metadata")
        return False, False, None, None, None

    comment = metadata["Comment"]
    if "signed_hash" not in comment:
        #raise RuntimeError("signed_hash not in comment")
        return False, False, None, None, None

    signed_hash = comment["signed_hash"].encode("utf-8")
    signed_hash = base64.b64decode(signed_hash)
    del comment["signed_hash"]

    verify_key_hex = verify_key_hex.encode("utf-8")
    verify_key = VerifyKey(verify_key_hex, encoder=Base64Encoder)

    good_latents, have_latents, latents = verify_latents(image, signed_hash, verify_key)
    if not good_latents:
        return False, False, None, None, None

    np_img = np.array(image)
    rgb = np_img[:, :, :3].tobytes()
    json_data = json.dumps(comment).encode("utf-8")
    image_and_comment = rgb + json_data
    fixed_png = None
    errs = 0

    try:
        verify_key.verify(image_and_comment, signed_hash)
    except:
        try:
            rgb, errs = fec_decode(bytearray(rgb), bytearray(fec_data), w, h)
            image_and_comment = rgb + json_data
            verify_key.verify(image_and_comment, signed_hash)
            if output_fixed:
                rgb = np.frombuffer(rgb, dtype=np.uint8)
                fixed_png = add_fec(image, rgb.reshape(np_img.shape[:2] + (3,)))
        except:
            return False, False, None, None, None

    return True, have_latents, errs, fixed_png, latents

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to image to check", required=True)
    parser.add_argument("-f", "--fixed-output", type=str, help="If error correction takes place and a filename is specified, the corrected RGB data is written to it")
    args = parser.parse_args()

    image_path = args.input
    print(f"Checking image at {image_path}")
    image = Image.open(image_path)
    import time
    t = time.perf_counter()
    is_novelai, have_latents, errs, fixed_png, latents = verify_image_is_novelai(image, output_fixed=args.fixed_output is not None)
    if errs is not None and errs > 0:
        print(f"Corrected pixel error bits: {errs}")
    print (f"Time taken: {time.perf_counter() - t:.4f}s")
    print(f"Is image NovelAI? {is_novelai}")
    if have_latents:
        print(f"Has latents. Width: {latents[0]}")

    if args.fixed_output is not None and fixed_png is not None:
        with open(args.fixed_output, "wb") as f:
            f.write(fixed_png)
        print(f"Fixed image written to {args.fixed_output}")
