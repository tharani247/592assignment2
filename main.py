import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os  # Importing os for file path operations
from scipy.spatial.distance import euclidean
from scipy.fftpack import fft
from scipy.stats import entropy
import pywt
import librosa.display

# Define dataset path
dataset_path = "Data/"  # Use the local folder inside PyCharm
mat_files = ['APPLIANCES.mat', 'HVAC.mat', 'LIGHTING.mat', 'MAINS.mat', 'OTHER_LOADS.mat']

def load_mat_file(file_path):
    """Load time series data from a .mat file"""
    data = scipy.io.loadmat(file_path)  # Load .mat file
    return data[list(data.keys())[-1]].flatten()  # Extract the time series array

def normalize_data(series):
    """Normalize time series data between 0 and 1"""
    return (series - np.min(series)) / (np.max(series) - np.min(series))

def moving_average(series, window_size=5):
    """Apply a simple moving average filter for denoising"""
    return np.convolve(series, np.ones(window_size)/window_size, mode='valid')

def compute_fft(series):
    """Compute the Fast Fourier Transform (FFT) of a time series"""
    n = len(series)
    freq = np.fft.fftfreq(n, d=1)  # Compute frequency bins
    fft_values = np.abs(fft(series))  # Compute absolute values of FFT components
    return freq[:n//2], fft_values[:n//2]  # Return half of the symmetric FFT output

def kl_divergence(p, q):
    """Compute Kullback-Leibler divergence between two probability distributions"""
    p = np.array(p) + 1e-10  # Avoid zero probabilities to prevent log issues
    q = np.array(q) + 1e-10
    return entropy(p, q)

def wavelet_transform(series, wavelet='db4', level=4):
    """Perform wavelet transform to extract time-frequency features"""
    coeffs = pywt.wavedec(series, wavelet, level=level)  # Compute wavelet decomposition
    return coeffs[0]  # Return approximation coefficients

def plot_spectrogram(series, sr=1):
    """Plot a spectrogram to detect motifs in the time series"""
    n_fft = min(len(series) // 2, 1024)  # Adjust FFT size to avoid warnings
    S = librosa.feature.melspectrogram(y=series, sr=sr, n_fft=n_fft)  # Compute mel spectrogram
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))  # Convert power spectrogram to dB scale
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram of Energy Usage (n_fft={n_fft})")

def show_all_plots():
    """Display all generated plots simultaneously"""
    plt.show()

# Load and preprocess all datasets
time_series_data = {}
for file in mat_files:
    full_path = os.path.join(dataset_path, file)  # Construct full file path
    series = load_mat_file(full_path)  # Load time series data from .mat file
    normalized_series = normalize_data(series)  # Normalize data between 0 and 1
    denoised_series = moving_average(normalized_series)  # Apply moving average filter
    time_series_data[file] = denoised_series  # Store processed data

# Example comparison between two time series using Euclidean distance
series1 = time_series_data['APPLIANCES.mat'][:1000]  # Extract first 1000 samples
series2 = time_series_data['HVAC.mat'][:1000]
euclidean_distance = euclidean(series1, series2)  # Compute Euclidean distance
print("Euclidean Distance:", euclidean_distance)

# Compute FFT and plot
target_series = time_series_data['MAINS.mat'][:1000]  # Extract first 1000 samples for FFT
freqs, fft_values = compute_fft(target_series)  # Compute FFT
plt.figure()
plt.plot(freqs, fft_values)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT of Power Consumption")

# Compute KL Divergence
kl_div = kl_divergence(series1, series2)  # Compute KL divergence between two series
print("KL Divergence:", kl_div)

# Perform Wavelet Transform and plot
target_wavelet = wavelet_transform(target_series)  # Compute wavelet transform
plt.figure()
plt.plot(target_wavelet)
plt.xlabel("Time")
plt.ylabel("Wavelet Coefficients")
plt.title("Wavelet Transform Features")

# Plot Spectrogram
plt.figure()
plot_spectrogram(target_series)

# Show all plots together
show_all_plots()
