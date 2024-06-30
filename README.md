This project is the python code for my bachelor's thesis mathematics & computer science

Requirements:

- Python 3.6+
- Standard python libraries, including os, sys and multiprocessing.
- Numpy
- Matplotlib
- PyWavelets
- Some parts of the code require SciPy, but this is not used in the normal workflow.

How to run:
Find a source of images and a test image, all of the same resolution.
This PRNU comparison is not scale or translation invariant, so cropping different parts of the image does not result in functional results.
Set the large window size global variable in main.py to a even value that divides both the horizontal and vertical resolution, somewhere between 120 and 400 is adviced.
Estimate the standard deviation of the white (Gaussian) noise component of these images and enter it.

Then, utilize run.py first in extract mode to extract the PRNU noise of every image.
The syntax is as follows:
run.py Extract [image_folder] [extension] [residue_output_folder]
Will extract the noise residue from all files that end with [extension] in image folder and place them into
identically named .npy files in residue output folder.
By default Extract will spawn multiple processes to speed up extraction.
Set the max_thread_count value in main.py to a value that is at least a little
lower than the logical thread count of your device.

Then compile the fingerprint of all images from the same sensor by using:
run.py Fingerprint [image_folder] [extension] [residue_folder] [fingerprint_path]
Will compile the images and residues given by image_folder and residue_folder into a fingerprint that will be sent to fingerprint_path.
Only images that end with extension will be used.
both the images and residues are sorted by the last integer in their filename, and matched in order.

Finially, match other extracted residues against the fingerprint using:
run.py Match [residue_path] [fingerprint_path]

There are also a number of demo scripts in the demo folder.
These are used mainly to generate figures used in the thesis.
