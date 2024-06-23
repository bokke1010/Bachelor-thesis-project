import main
import matplotlib.pyplot as plt

cat = main.load_image("muis_small.png")

main.large_window_size = 256


(image, _) = main.denoise_full(cat)

plt.imshow(image / 255.0)
plt.show()

