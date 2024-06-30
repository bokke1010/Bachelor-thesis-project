import numpy as np
import matplotlib.pyplot as plt
import pywt


res = 100
time = 6
x = np.linspace(0, time, time * res, endpoint=False)

# wavelet = "db9"
# wavelet = "rbio3.1"
wavelet = "cmor1.5-1.0"

fig, axs = plt.subplots(2, 2, sharex=True)

# Constant frequency
if True:

    oy = 0.6 * np.cos(5*np.pi * x / 2.0) + 0.4 * np.cos(23*np.pi * x / 2.0)
    
    axs[0,0].set_xlabel("Time")
    axs[0,0].set_ylabel("y")
    axs[0,0].plot(x,oy)

    widths = np.geomspace(1, 1024, num=100)
    sampling_period = np.diff(x).mean()
    cwtmatr, freqs = pywt.cwt(oy, widths, wavelet, sampling_period=sampling_period)
    # absolute take absolute value of complex result
    cwtmatr = np.abs(cwtmatr[:-1, :-1])

    pcm = axs[0,1].pcolormesh(x, freqs, cwtmatr)
    axs[0,1].set_yscale("log")
    axs[0,1].set_xlabel("Time (s)")
    axs[0,1].set_ylabel("Frequency (Hz)")
    # axs[0,1].set_title("Continuous Wavelet Transform (Scaleogram)")
    fig.colorbar(pcm, ax=axs[0,1])


# Dynamic frequency
if True:

    y = np.empty_like(x)
    y[:2*res] = np.cos(3*np.pi * x[:2*res] / 2.0)
    y[2*res:4*res] = -np.cos(8*np.pi * x[2*res:4*res])
    y[4*res:5*res] = -np.cos(16 * np.pi * x[4*res:5*res])
    y[5*res:] = -np.cos(32 * np.pi * x[5*res:])

    
    axs[1,0].set_xlabel("Time")
    axs[1,0].set_ylabel("y")
    axs[1,0].plot(x,y)

    widths = np.geomspace(1, 1024, num=100)
    sampling_period = np.diff(x).mean()
    cwtmatr, freqs = pywt.cwt(y, widths, wavelet, sampling_period=sampling_period)
    # absolute take absolute value of complex result
    cwtmatr = np.abs(cwtmatr[:-1, :-1])

    pcm = axs[1,1].pcolormesh(x, freqs, cwtmatr)
    axs[1,1].set_yscale("log")
    axs[1,1].set_xlabel("Time (s)")
    axs[1,1].set_ylabel("Frequency (Hz)")
    # axs[1,1].set_title("Continuous Wavelet Transform (Scaleogram)")
    fig.colorbar(pcm, ax=axs[1,1])

plt.suptitle("The continuous wavelet transform for stationary and time-dependant signals.")
plt.show()
