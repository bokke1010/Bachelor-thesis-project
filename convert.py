import os, sys
from PIL import Image

def convert(path, x, y, extension = ".JPG"):
    """Overrides all JPG files in path with rescaled versions with resolution (x, y).
    Uses LANCZOS resampling."""
    
    count, total = 0,0

    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file and entry.name.endswith(extension):
                im = Image.open(entry)
                im = im.resize((x, y), Image.Resampling.LANCZOS)
                im.save(entry)
                count += 1
            total += 1
    print(f"Rescaled {count}/{total} files in the given folder.")

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Convert takes 3 arguments: x, y, path, [extension]")
        exit()
    path = sys.argv[3]
    try:
        x, y = int(sys.argv[1]), int(sys.argv[2])
    except:
        raise TypeError("Could not interpret argument types.")
    
    if len(sys.argv) == 5:
        convert(path, x, y, sys.argv[4])
    else:
        convert(path, x, y)