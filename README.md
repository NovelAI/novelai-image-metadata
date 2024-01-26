# NovelAI Image Metadata Scripts

Meta data extraction scripts for images generated with NovelAI's image generation functionality.

`nai_meta.py` extracts prompt information and other settings from the alpha channel of NAI generated images. The information is stored in the stealth pnginfo format.

`nai_sig.py` will verify the signature in the `signed_hash` field, which allows us to verify whether an image was generated by our models. No user specific data is included in this field. If necessary and such information is present in the alpha channel, it will perform error correction on the signed RGB data. The corrected image can be saved using the `-f` argument.

`nai_meta_writer.py` allows you to write the metadata to the alpha channel of an image. This is useful if you want to add metadata to an image that was not generated by NovelAI. It will also be added as regular PNG meta data. In addition, forward error correction (FEC) codes for the RGB data are generated and also injected into the alpha channel. This allows you to recover the original image even if the RGB data is slightly damaged.

`nai_add_fec.py` adds FEC data to an image, preserving metadata in the alpha channel and in the PNG meta data.

## Requirements

Install requirements like this:

```bash
pip install pynacl pydantic bchlib
```

## Why forward error correction (FEC)?

We have found that when copying images to the clipboard in certain browsers, a few pixels will get modified very slightly, which will cause our signature check to fail. Since direct copy and paste is often more convenient than saving the image first, this will become a usability issue in the future. The signature lets us identify images that were generated by our models. In the future, there will be functionality for users that is only available for images that were generated by our models. This is why we want to be able to recover the original image even if a few pixels are damaged.