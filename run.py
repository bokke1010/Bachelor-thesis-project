# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# The main script to utilize the functionality within the other scripts.
# Can be used to extract PRNU noise, calculate fingerprints and match
# multiple PRNU noise residues against a fingerprint.

import main
import os
import numpy as np
import sys
import os
import re
from tools.np_imageload import load_image, save_image, save_image_grayscale

helpmessage = """This script can extract PRNU noise residues, compile a fingerprint and test images against a fingerprint.
Usage:
run.py Extract [image_folder] [extension] [residue_output_folder]
    Will extract the noise residue from all files in image folder and place them into identically named .npy files in residue output folder.
    Only images that end with extension will be used.\n
run.py Fingerprint [image_folder] [extension] [residue_folder] [fingerprint_path]
    Will compile the images and residues given by image_folder and residue_folder into a fingerprint that will be sent to fingerprint_path.
    Only images that end with extension will be used.\n
run.py Match [image_path] [fingerprint_path]
    Will calculate the PCE between the image noise residue and the fingerprint.
"""

def run(mode, args):    
    # Modes is one of "Extract", "Fingerprint", "Match"

    if mode in ["Help", "help", "--help", "-h"]:
        print(helpmessage)

    elif mode == "Extract":

        if len(args) < 3:
            print("Extract mode must have an input folder path, file extension and output folder path argument.\nFile extention may include additional characters to filter files.")
            return

        image_path = args[0]
        extension = args[1]
        residue_path = args[2]

        if not os.path.isdir(image_path):
            print("Given input path is not a valid directory, aborting.")
            return

        if not os.path.isdir(residue_path):
            print("Given output path is not a valid directory, aborting.")
            return

        images = []
        filenames = []
        with os.scandir(image_path) as it:
            for entry in it:
                if entry.is_file and entry.name.endswith(extension):
                    images.append(load_image(entry.path))
                    filenames.append(os.path.join(residue_path, entry.name.split('.')[0] + ".npy"))

        if len(images) == 0:
            print("No images found, aborting.")
            return

        fingerprint = main.extract_residues(images, filenames, True)
        

    elif mode == "Fingerprint":
        if len(args) < 4:
            print("Fingerprint mode must have an image input folder path, extention, residue input folder path and output file path argument.\nFile extention may include additional characters to filter files.")
            return

        image_path = args[0]
        image_extension = args[1]
        residue_path = args[2]
        fingerprint_path = args[3]

        if not os.path.isdir(image_path):
            print("Given image path is not a valid directory, aborting.")
            return

        if not os.path.isdir(image_path):
            print("Given residue path is not a valid directory, aborting.")
            return
        
        # if not os.path.isfile(fingerprint):
        #     print("Given output path is not a valid filepath.")

        image_pairs = []
        residue_pairs = []
        with os.scandir(image_path) as it:
            for entry in it:
                if entry.is_file and entry.name.endswith(image_extension):
                    image_digit = int(re.search(r'(\d+)\D*$', entry.name).group(1))
                    image_pairs.append((image_digit, load_image(entry.path)))


        with os.scandir(residue_path) as it:
            for entry in it:
                if entry.is_file and entry.name.endswith(".npy"):
                    with open(entry.path, "rb") as f:
                        residue_digit = int(re.search(r'(\d+)\D*$', entry.name).group(1))
                        residue_pairs.append((residue_digit, np.load(f)))

        images = [a[1] for a in sorted(image_pairs)]
        residues = [a[1] for a in sorted(residue_pairs)]

        assert len(images) > 0 and len(images) == len(residues)

        fingerprint = main.find_fingerprint(images, residues)

        with open(fingerprint_path, "wb") as f:
            fingerprint = np.save(f, fingerprint)

    elif mode == "Match":
        if len(args) < 2:
            print("Match mode must have a residue folder path and fingerprint path argument, aborting.")
            return

        residue_path = args[0]
        fingerprint_path = args[1]

        fingerprint = None
        with open(fingerprint_path, 'rb') as f:
            fingerprint = np.load(f)

        residues, names = [], []

        with os.scandir(residue_path) as it:
            for entry in it:
                if entry.is_file and entry.name.endswith(".npy"):
                    with open(entry.path, "rb") as f:
                        residues.append(np.load(f))
                        names.append(entry.name)

        assert len(residues) > 0 and len(residues) == len(names)

        main.test_fingerprint_SPCE_multiple(fingerprint, residues, names)
    else:
        print("mode must be one of \"Help\", \"Extract\", \"Fingerprint\" or \"Match\".")

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print(helpmessage)
    else:
        run(sys.argv[1], sys.argv[2:])
