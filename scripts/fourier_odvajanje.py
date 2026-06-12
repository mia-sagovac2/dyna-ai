import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

y, sr = librosa.load("../data/sorted_label_studio/BIC/5ceeff0b-2023-08-04_10-25-40.0-Swanston6_TJN_NoV_L1_1Lane_Idling_mono.wav", sr=None)
os.makedirs("../fourier_separated_audio", exist_ok=True)

# HPSS
y_harmonic, y_percussive = librosa.effects.hpss(y)

# STFT za vizualizaciju
D_orig = librosa.stft(y)
D_harm = librosa.stft(y_harmonic)
D_perc = librosa.stft(y_percussive)

# pretvori u dB
S_orig = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)
S_harm = librosa.amplitude_to_db(np.abs(D_harm), ref=np.max)
S_perc = librosa.amplitude_to_db(np.abs(D_perc), ref=np.max)

# plot
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
librosa.display.specshow(S_orig, sr=sr, x_axis='time', y_axis='hz')
plt.title("Original")

plt.subplot(3, 1, 2)
librosa.display.specshow(S_harm, sr=sr, x_axis='time', y_axis='hz')
plt.title("Harmonic")

plt.subplot(3, 1, 3)
librosa.display.specshow(S_perc, sr=sr, x_axis='time', y_axis='hz')
plt.title("Percussive")


# generirati audio fileove za pregled
plt.savefig("../fourier_separated_audio/spectrograms.png")
sf.write("../fourier_separated_audio/harmonic.wav", y_harmonic, sr)
sf.write("../fourier_separated_audio/percussive.wav", y_percussive, sr)