import main
import os
import numpy as np

if __name__ == '__main__':
    image = main.load_image("muis_small.png")
    
    mode = "Extract"
    image_path = "B:\\Images PRNU\\Onderzoek naar foto-video-vergelijkingenlars\\Onderzoek naar foto-video-vergelijkingen\\All Images\\iPhone 8 Plus\\iPhone8Plus_Photos\\iPhone8Plus_natural_noHDR"
    extension = ").JPG"


    if mode == "Extract":
        images = []
        with os.scandir(image_path) as it:
            for entry in it:
                if entry.is_file and entry.name.endswith(extension):
                    images.append(main.load_image(entry.path))

        fingerprint = main.find_fingerprint(images[:4])
        with open('fingerprint.npy', 'wb') as f:
            np.save(f, fingerprint)

    if mode == "Match":
        fingerprint = None
        with open('fingerprint.npy', 'rb') as f:
            fingerprint = np.load(f)

        image = main.load_image(image_path)
        (residue, _) = main.denoise_full(image)
        corr = main.test_fingerprint_SPE(fingerprint, residue)
        print(corr)
