import numpy as np

# spectrum: 2D array of shape T x F (T is # of windows and F is # of features)
def spect2raw(spectrum):
	data = []
	for freq_data in spectrum:
		time_data = np.fft.ifft(freq_data)
		data.append(time_data)
	data = np.concatenate(data)
	return data

# raw: 1D array
# time_frame: window size on which to compute FFT
def raw2spect(raw, time_frame):
	num_intervals = len(raw) // time_frame
	spectrum = []
	for i in range(num_intervals):
		time_data = raw[time_frame*i:time_frame*(i+1)]
		spectrum.append(np.fft.fft(time_data))
	return spectrum