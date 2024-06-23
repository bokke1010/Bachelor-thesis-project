import os, sys
from PIL import Image

size = (1008, 756)

image_path = "B:\\Images PRNU\\Onderzoek naar foto-video-vergelijkingenlars\\Onderzoek naar foto-video-vergelijkingen\\All Images\\other_images_small"


with os.scandir(image_path) as it:
    for entry in it:
        if entry.is_file and entry.name.endswith(".JPG"):
            im = Image.open(entry)
            im = im.resize(size, Image.Resampling.LANCZOS)
            im.save(entry)
