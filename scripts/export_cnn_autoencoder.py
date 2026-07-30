"""Trenira finalne verzije CNN i autoencoder binarnih klasifikatora (sum vs. vozilo) -
identican pipeline kao u notebookovima `notebooks/03_modeli/cnn_klasifikator.ipynb` i
`notebooks/03_modeli/autoencoder_klasifikator.ipynb` - i exporta ih u ONNX format u
mapu modeli_exports/binary_classification/.

Modeli:
  1. CNN nad spektrogram slikama (Conv2D x4 -> Dense) - cnn_klasifikator.ipynb
  2. Autoencoder nad feature-vektorom (25 znacajki)    - autoencoder_klasifikator.ipynb

Za autoencoder se uz ONNX model exporta i StandardScaler (mean/scale po znacajki),
threshold (Youden's J) i redoslijed znacajki na ulazu - kontroler mora sam raditi
skaliranje ulaza, racunati MSE rekonstrukcije i usporediti s thresholdom (ONNX graf
sadrzi samo encoder/decoder mrezu, ne i tu logiku).

Pokretanje:  python scripts/export_cnn_autoencoder.py  (iz roota repoa)

Ovisnosti uz requirements.txt: tf2onnx (za konverziju Keras modela u ONNX).
"""

import os
import glob

# tf2onnx ne podrzava Keras 3 (default backend od TF 2.16+) - reverse_lookup u
# tf2onnx/convert.py:from_keras puca s KeyError jer se imena izlaznih tenzora
# vise ne poklapaju. Prisiljavamo legacy Keras 2 API (paket tf-keras) prije
# importa tensorflowa, jer to tf2onnx ocekuje.
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.cm as cm
import librosa
from scipy import signal
from scipy.stats import linregress

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

import tf2onnx
import onnxruntime as ort

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_DIR = os.path.join(BASE, 'modeli_exports', 'binary_classification')
os.makedirs(EXPORT_DIR, exist_ok=True)


def verify_onnx(path, keras_model, X_check, atol=1e-4):
    """Provjeri da ONNX model daje (skoro) identicne izlaze kao originalni Keras model."""
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: X_check.astype(np.float32)})[0]
    keras_out = keras_model.predict(X_check, verbose=0)
    max_diff = np.max(np.abs(onnx_out.reshape(keras_out.shape) - keras_out))
    size_kb = os.path.getsize(path) / 1024
    print(f"  -> {os.path.basename(path)} ({size_kb:.0f} KB), "
          f"max apsolutna razlika ONNX vs Keras: {max_diff:.2e} "
          f"({'OK' if max_diff < atol else 'PROVJERI'})")
    return max_diff


def export_keras_to_onnx(model, input_shape, onnx_path):
    spec = (tf.TensorSpec((None,) + input_shape, tf.float32, name='input'),)
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)


# ======================================================================
# 1. CNN klasifikator nad spektrogram slikama - cnn_klasifikator.ipynb
# ======================================================================
print("[1/2] CNN klasifikator (spektrogrami)...")

RAW_BG_DIR = os.path.join(BASE, 'data', 'raw', 'background')
RAW_VEH_DIR = os.path.join(BASE, 'data', 'raw', 'vehicle')
SPEC_BG_DIR = os.path.join(BASE, 'data', 'spectrograms', 'BG')
SPEC_VEH_DIR = os.path.join(BASE, 'data', 'spectrograms', 'Veh')

FRAME_SIZE = 512
HOP_SIZE = 64
N_MELS = 64


def normalize_audio(audio):
    return 2 * ((audio - np.min(audio)) / np.ptp(audio)) - 1


def audio_to_spectrogram_image(path):
    y, sr = librosa.load(path, sr=None)
    y = y[:sr * 2]
    y = normalize_audio(y)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    norm = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-9)
    rgb = (cm.magma(norm)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def generate_spectrograms(src_dir, dst_dir, label):
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]
    existing = len([f for f in os.listdir(dst_dir) if f.endswith('.png')])
    if existing >= len(files):
        print(f"  {label}: vec generirano ({existing} slika), preskacem")
        return
    for i, fname in enumerate(files):
        out_path = os.path.join(dst_dir, fname.replace('.wav', '.png'))
        if os.path.exists(out_path):
            continue
        img = audio_to_spectrogram_image(os.path.join(src_dir, fname))
        img.save(out_path)
        if i % 2000 == 0:
            print(f"  {label}: {i}/{len(files)}")
    print(f"  {label}: gotovo ({len(files)} slika)")


generate_spectrograms(RAW_BG_DIR, SPEC_BG_DIR, 'background')
generate_spectrograms(RAW_VEH_DIR, SPEC_VEH_DIR, 'vehicle')

bg_files = glob.glob(os.path.join(SPEC_BG_DIR, '*.png'))
veh_files = glob.glob(os.path.join(SPEC_VEH_DIR, '*.png'))
paths = bg_files + veh_files
labels = [0] * len(bg_files) + [1] * len(veh_files)  # 0 = background (sum), 1 = vehicle
print(f"  background: {len(bg_files)}, vehicle: {len(veh_files)}, ukupno: {len(paths)}")

paths_train, paths_test, y_train, y_test = train_test_split(
    paths, labels, test_size=0.15, stratify=labels, random_state=42)
paths_train, paths_val, y_train, y_val = train_test_split(
    paths_train, y_train, test_size=0.15, stratify=y_train, random_state=42)

IMG_HEIGHT, IMG_WIDTH = 163, 223
BATCH_SIZE = 64


def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(paths_train, y_train, shuffle=True)
val_ds = make_dataset(paths_val, y_val)
test_ds = make_dataset(paths_test, y_test)

tf.keras.utils.set_random_seed(42)

cnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])


def step_decay(epoch, lr):
    initial_lr = 0.0001
    drop = 0.1
    epochs_drop = 25
    return initial_lr * (drop ** (epoch // epochs_drop))


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=0)
early_stop_cnn = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True,
)

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[lr_scheduler, early_stop_cnn],
    verbose=2,
)

test_loss, test_acc = cnn.evaluate(test_ds, verbose=0)
print(f"  Test accuracy: {test_acc:.3f}")

# Spremi istrenirani model prije ONNX exporta - ako export padne, ne treba
# ponovno trenirati nego samo ucitati ovo i pokusati export ponovno.
cnn.save(os.path.join(EXPORT_DIR, 'cnn_klasifikator.keras'))

cnn_path = os.path.join(EXPORT_DIR, 'cnn_klasifikator.onnx')
export_keras_to_onnx(cnn, (IMG_HEIGHT, IMG_WIDTH, 3), cnn_path)

X_test_batch = np.concatenate([x.numpy() for x, _ in test_ds.take(1)])
verify_onnx(cnn_path, cnn, X_test_batch)

# ======================================================================
# 2. Autoencoder klasifikator nad feature-vektorom - autoencoder_klasifikator.ipynb
# ======================================================================
print("\n[2/2] Autoencoder klasifikator (feature-vektor)...")

BG_PATH = os.path.join(BASE, 'data', 'raw', 'background')
VEH_PATH = os.path.join(BASE, 'data', 'raw', 'vehicle')
CACHE_PATH = os.path.join(BASE, 'data', 'features_cache', 'znacajke_cache_bg_veh.pkl')


def remove_mains_hum(y, sr, hum_freqs=(50, 100, 150, 200), Q=30):
    y_filt = y.copy()
    for f0 in hum_freqs:
        if f0 >= sr / 2:
            continue
        b, a = signal.iirnotch(f0, Q, sr)
        y_filt = signal.filtfilt(b, a, y_filt)
    return y_filt


def highpass(y, sr, cutoff=15, order=4):
    sos = signal.butter(order, cutoff, btype='highpass', fs=sr, output='sos')
    return signal.sosfiltfilt(sos, y)


def preprocess_audio(y, sr):
    y = remove_mains_hum(y, sr)
    y = highpass(y, sr)
    return y


LOW_BAND = (0, 300)
MID_BAND = (300, 2000)
HIGH_BAND = (2000, 11025)


def fft_psd(y, sr):
    window = np.hanning(len(y))
    y_win = y * window
    fft = np.fft.rfft(y_win)
    psd = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    return freqs, psd


def band_energy_ratios(freqs, psd):
    total = psd.sum() + 1e-12

    def band_frac(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return psd[mask].sum() / total

    return band_frac(*LOW_BAND), band_frac(*MID_BAND), band_frac(*HIGH_BAND)


def spectral_slope(freqs, psd, f_min=20, f_max=4000):
    mask = (freqs >= f_min) & (freqs <= f_max) & (psd > 0)
    if mask.sum() < 2:
        return 0.0
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    slope, _, _, _, _ = linregress(log_f, log_p)
    return slope


def bandwidth_low_band(freqs, psd):
    lo, hi = LOW_BAND
    mask = (freqs >= lo) & (freqs < hi)
    f = freqs[mask]
    p = psd[mask]
    if p.sum() <= 0:
        return 0.0
    centroid = np.average(f, weights=p)
    return np.sqrt(np.average((f - centroid) ** 2, weights=p))


def extract_features(y, sr):
    freqs, psd = fft_psd(y, sr)

    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    low_e, mid_e, high_e = band_energy_ratios(freqs, psd)
    slope = spectral_slope(freqs, psd)
    bw_low = bandwidth_low_band(freqs, psd)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    flux = np.abs(np.diff(onset_env))

    flatness = librosa.feature.spectral_flatness(y=y)[0]

    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.sum(y_harm ** 2)
    perc_energy = np.sum(y_perc ** 2)
    harmonic_ratio = harm_energy / (harm_energy + perc_energy + 1e-12)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)

    feats = {
        'rms': rms,
        'zcr': zcr,
        'spectral_centroid': centroid,
        'spectral_bandwidth': bandwidth,
        'spectral_rolloff': rolloff,
        'low_band_energy': low_e,
        'mid_band_energy': mid_e,
        'high_band_energy': high_e,
        'spectral_slope': slope,
        'bandwidth_low_band': bw_low,
        'flux_mean': np.mean(flux),
        'flux_std': np.std(flux),
        'spectral_flatness_mean': np.mean(flatness),
        'spectral_flatness_std': np.std(flatness),
        'harmonic_ratio': harmonic_ratio,
    }
    for i in range(5):
        feats[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
        feats[f'mfcc_{i}_std'] = np.std(mfcc[i])
    return feats


if os.path.exists(CACHE_PATH):
    df = pd.read_pickle(CACHE_PATH)
    print(f"  Znacajke ucitane iz cachea: {CACHE_PATH} ({len(df)} uzoraka)")
else:
    data = []
    for folder, label in [(BG_PATH, 'background'), (VEH_PATH, 'vehicle')]:
        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        for i, file in enumerate(files):
            path = os.path.join(folder, file)
            if i % 500 == 0:
                print(f"   {label}: {i}/{len(files)} -> {file}")
            y, sr = librosa.load(path, sr=None)
            y = y[:sr * 2]
            y = preprocess_audio(y, sr)
            feats = extract_features(y, sr)
            feats['file'] = file
            feats['label'] = label
            data.append(feats)
    df = pd.DataFrame(data)
    df = df[[c for c in df.columns if c not in ('file', 'label')] + ['file', 'label']]
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_pickle(CACHE_PATH)
    print(f"  Znacajke spremljene u cache: {CACHE_PATH}")

feature_cols = [c for c in df.columns if c not in ('file', 'label')]

df_bg = df[df['label'] == 'background'][feature_cols].reset_index(drop=True)
df_veh = df[df['label'] == 'vehicle'][feature_cols].reset_index(drop=True)

bg_train, bg_temp = train_test_split(df_bg, test_size=0.4, random_state=42)
bg_val, bg_test = train_test_split(bg_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_bg_train = scaler.fit_transform(bg_train)
X_bg_val = scaler.transform(bg_val)
X_bg_test = scaler.transform(bg_test)
X_veh_test = scaler.transform(df_veh)

print(f"  Background - train: {len(bg_train)}, val: {len(bg_val)}, test: {len(bg_test)}")
print(f"  Vehicle (samo za evaluaciju/threshold): {len(df_veh)}")

tf.keras.utils.set_random_seed(42)

n_features = X_bg_train.shape[1]

autoencoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_features,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),   # bottleneck
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(n_features, activation='linear'),
])
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')

early_stop_ae = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True,
)

autoencoder.fit(
    X_bg_train, X_bg_train,
    validation_data=(X_bg_val, X_bg_val),
    epochs=200, batch_size=64, callbacks=[early_stop_ae], verbose=0,
)


def reconstruction_error(model, X):
    X_hat = model.predict(X, verbose=0)
    return np.mean((X - X_hat) ** 2, axis=1)


err_bg_test = reconstruction_error(autoencoder, X_bg_test)
err_veh_test = reconstruction_error(autoencoder, X_veh_test)

# Threshold: Youden's J statistic na ROC krivulji (bg_test vs. veh_test) - isti zakljucak
# kao u autoencoder_klasifikator.ipynb (95. percentil na background train gresci je losiji).
y_true = np.concatenate([np.zeros(len(err_bg_test)), np.ones(len(err_veh_test))])
y_score = np.concatenate([err_bg_test, err_veh_test])
fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
j_scores = tpr - fpr
threshold = float(roc_thresholds[np.argmax(j_scores)])
print(f"  Threshold (Youden's J): {threshold:.4f}")

autoencoder.save(os.path.join(EXPORT_DIR, 'autoencoder_klasifikator.keras'))

ae_path = os.path.join(EXPORT_DIR, 'autoencoder_klasifikator.onnx')
export_keras_to_onnx(autoencoder, (n_features,), ae_path)
verify_onnx(ae_path, autoencoder, X_bg_test[:256])

# Autoencoder ONNX sadrzi samo encoder/decoder mrezu - kontroler mora sam raditi
# skaliranje ulaza, racunati MSE(ulaz, rekonstrukcija) i usporediti s thresholdom.
scaler_path = os.path.join(EXPORT_DIR, 'autoencoder_scaler.txt')
with open(scaler_path, 'w') as f:
    f.write("# znacajka mean scale(std)\n")
    for col, mean, scale in zip(feature_cols, scaler.mean_, scaler.scale_):
        f.write(f"{col} {mean} {scale}\n")

threshold_path = os.path.join(EXPORT_DIR, 'autoencoder_threshold.txt')
with open(threshold_path, 'w') as f:
    f.write(f"{threshold}\n")

features_path = os.path.join(EXPORT_DIR, 'autoencoder_redoslijed_znacajki.txt')
with open(features_path, 'w') as f:
    for i, col in enumerate(feature_cols):
        f.write(f"{i} {col}\n")

print(f"\nGotovo! Sve spremljeno u: {EXPORT_DIR}")
for name in sorted(os.listdir(EXPORT_DIR)):
    print(f"  - {name}")
