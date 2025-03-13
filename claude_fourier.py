import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.fftpack import fft, ifft, fftshift, fftfreq

# Set up a clean style for plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 10})

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig)

# Parameters
duration = 1.0  # signal duration in seconds
fs = 1000       # sampling frequency in Hz
t = np.linspace(0, duration, int(fs*duration), endpoint=False)  # time vector

# Create a complex signal with multiple frequency components
f1, f2, f3 = 5, 15, 40  # Hz
signal = 5*np.sin(2*np.pi*f1*t) + 3*np.sin(2*np.pi*f2*t) + np.sin(2*np.pi*f3*t)

# Plot 1: Original time domain signal
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, signal, 'b-')
ax1.set_title('Original Time Domain Signal (Sum of 5 Hz, 15 Hz, and 40 Hz components)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')

# Calculate Fourier Transform
X = fft(signal)
freqs = fftfreq(len(t), 1/fs)
X_mag = np.abs(X) / len(t)  # Normalize

# Plot 2: Frequency domain representation
ax2 = fig.add_subplot(gs[1, :])
ax2.stem(freqs[:len(freqs)//2], 2*X_mag[:len(X_mag)//2], 'r', markerfmt='ro', basefmt='-')
ax2.set_title('Frequency Domain Representation (Fourier Transform)')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_xlim(0, 50)  # Limit x-axis to the relevant frequencies

# Function to plot basis functions
def plot_basis(ax, freq, title):
    basis = np.sin(2*np.pi*freq*t)
    ax.plot(t[:100], basis[:100], 'g-')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    return basis

# Plot basis functions
ax3 = fig.add_subplot(gs[2, 0])
basis1 = plot_basis(ax3, f1, f'Basis Function: {f1} Hz')

ax4 = fig.add_subplot(gs[2, 1])
basis2 = plot_basis(ax4, f2, f'Basis Function: {f2} Hz')

ax5 = fig.add_subplot(gs[2, 2])
basis3 = plot_basis(ax5, f3, f'Basis Function: {f3} Hz')

# Calculate and display correlation matrix and covariance
def inner_product(a, b):
    return np.sum(a * b) / len(a)

correlation_matrix = np.zeros((3, 3))
basis_functions = [basis1, basis2, basis3]
labels = [f'{f1} Hz', f'{f2} Hz', f'{f3} Hz']

for i in range(3):
    for j in range(3):
        correlation_matrix[i, j] = inner_product(basis_functions[i], basis_functions[j])

# Create a text box with the correlation matrix
textbox_content = "Correlation Matrix Between Basis Functions:\n"
textbox_content += f"{'':10s} {labels[0]:10s} {labels[1]:10s} {labels[2]:10s}\n"
for i in range(3):
    textbox_content += f"{labels[i]:10s} "
    for j in range(3):
        textbox_content += f"{correlation_matrix[i, j]:10.5f} "
    textbox_content += "\n"
textbox_content += "\nNote: Values very close to 0 show orthogonality (no correlation)\n"
textbox_content += "between different frequencies."

# Add text box to figure
fig.text(0.1, 0.01, textbox_content, fontsize=10, 
         bbox=dict(facecolor='lightgray', alpha=0.5))

# Demonstrate filtering in frequency domain
fig2 = plt.figure(figsize=(16, 10))
gs2 = GridSpec(3, 2, figure=fig2)

# Original signal and spectrum
ax6 = fig2.add_subplot(gs2[0, 0])
ax6.plot(t, signal, 'b-')
ax6.set_title('Original Signal')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Amplitude')

ax7 = fig2.add_subplot(gs2[0, 1])
ax7.stem(freqs[:len(freqs)//2], 2*X_mag[:len(X_mag)//2], 'r', markerfmt='ro', basefmt='-')
ax7.set_title('Original Frequency Spectrum')
ax7.set_xlabel('Frequency (Hz)')
ax7.set_ylabel('Magnitude')
ax7.set_xlim(0, 50)

# Create filtered versions in frequency domain
X_filtered1 = X.copy()
X_filtered2 = X.copy()

# Filter 1: Remove 15 Hz component
mask = np.ones_like(X_filtered1, dtype=bool)
mask[(freqs >= 14) & (freqs <= 16)] = False
mask[(freqs >= -16) & (freqs <= -14)] = False  # Also remove negative frequency
X_filtered1[~mask] = 0

# Filter 2: Keep only 5 Hz component
mask = np.zeros_like(X_filtered2, dtype=bool)
mask[(freqs >= 4) & (freqs <= 6)] = True
mask[(freqs >= -6) & (freqs <= -4)] = True  # Also keep negative frequency
X_filtered2 = X_filtered2 * mask

# Convert back to time domain
signal_filtered1 = np.real(ifft(X_filtered1))
signal_filtered2 = np.real(ifft(X_filtered2))

# Plot filtered signals
ax8 = fig2.add_subplot(gs2[1, 0])
ax8.plot(t, signal_filtered1, 'g-')
ax8.set_title('Signal with 15 Hz Component Removed')
ax8.set_xlabel('Time (s)')
ax8.set_ylabel('Amplitude')

ax9 = fig2.add_subplot(gs2[1, 1])
X_filtered1_mag = np.abs(X_filtered1) / len(t)
ax9.stem(freqs[:len(freqs)//2], 2*X_filtered1_mag[:len(X_filtered1_mag)//2], 'g', markerfmt='go', basefmt='-')
ax9.set_title('Filtered Frequency Spectrum (15 Hz Removed)')
ax9.set_xlabel('Frequency (Hz)')
ax9.set_ylabel('Magnitude')
ax9.set_xlim(0, 50)

ax10 = fig2.add_subplot(gs2[2, 0])
ax10.plot(t, signal_filtered2, 'm-')
ax10.set_title('Signal with Only 5 Hz Component')
ax10.set_xlabel('Time (s)')
ax10.set_ylabel('Amplitude')

ax11 = fig2.add_subplot(gs2[2, 1])
X_filtered2_mag = np.abs(X_filtered2) / len(t)
ax11.stem(freqs[:len(freqs)//2], 2*X_filtered2_mag[:len(X_filtered2_mag)//2], 'm', markerfmt='mo', basefmt='-')
ax11.set_title('Filtered Frequency Spectrum (Only 5 Hz)')
ax11.set_xlabel('Frequency (Hz)')
ax11.set_ylabel('Magnitude')
ax11.set_xlim(0, 50)

fig2.text(0.5, 0.01, 
         "Fourier Transform allows us to modify frequency components independently\n"
         "because they are orthogonal (have zero covariance).", 
         fontsize=12, ha='center', 
         bbox=dict(facecolor='lightgray', alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.show()