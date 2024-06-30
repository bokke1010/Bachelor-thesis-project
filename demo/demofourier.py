

import numpy as np
import matplotlib.pyplot as plt


res = 100
x = np.linspace(0, 6, 6 * res, endpoint=False)

# Constant frequency

oy = 0.6 * np.cos(5*np.pi * x / 2.0) + 0.4 * np.cos(23*np.pi * x / 2.0)
plt.subplot(2,2,1)
plt.xlabel("Time")
plt.ylabel("y")
plt.plot(x,oy)

sp = np.fft.fft(oy)
frac = np.fft.fftfreq(6*res)
N = 160 # Shows only a tenth of the theoretical resolution
plt.subplot(2,2,3)
plt.plot(2 * res * frac[:N], np.abs(sp[:N]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Absolute frequency magnitude")

# Dynamic frequency

y = np.empty_like(x)
y[:2*res] = np.cos(3*np.pi * x[:2*res] / 2.0)
y[2*res:4*res] = -np.cos(8*np.pi * x[2*res:4*res])
y[4*res:5*res] = -np.cos(16 * np.pi * x[4*res:5*res])
y[5*res:] = -np.cos(32 * np.pi * x[5*res:])

plt.subplot(2,2,2)
plt.xlabel("Time")
plt.ylabel("y")
plt.plot(x,y)

sp = np.fft.fft(y)
frac = np.fft.fftfreq(6*res)
N = 160 # Shows only a tenth of the theoretical resolution
plt.subplot(2,2,4)
plt.plot(2 * res * frac[:N],np.abs(sp[:N]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Absolute frequency magnitude")

plt.suptitle("The Fourier transform for stationary and time-dependant signals.")
plt.show()
