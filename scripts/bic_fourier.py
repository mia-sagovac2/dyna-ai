import warnings
warnings.filterwarnings("ignore")
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt
from matplotlib.patches import Patch

FILE_PATH = "../data/all_sorted/BIC/5ceeff0b-2023-08-04_10-25-40.0-Swanston6_TJN_NoV_L1_1Lane_Idling_mono.wav"
OUTPUT_DIR = Path("../fourier_analiza")
SR         = 22050
DURATION   = 2.0

FREQ_ZONES = {
    "Sub-bass\n(20-60 Hz)":   (20,    60),
    "Bass\n(60-250 Hz)":      (60,   250),
    "Low-mid\n(250-500 Hz)":  (250,  500),
    "Mid\n(500-2k Hz)":       (500,  2000),
    "High-mid\n(2k-6k Hz)":  (2000, 6000),
    "High\n(6k-20k Hz)":     (6000, 20000),
}
ZONE_COLORS = ["#2c3e50", "#2980b9", "#27ae60", "#f39c12", "#e74c3c", "#9b59b6"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# helperi
def load_audio(path=FILE_PATH, sr=SR, duration=DURATION):
    y, _ = librosa.load(str(path), sr=sr, duration=duration, mono=True)
    target = int(sr * duration)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    return y

def compute_fft(y, sr=SR):
    N      = len(y)
    win    = np.hanning(N)
    Y      = np.fft.rfft(y * win)
    freqs  = np.fft.rfftfreq(N, d=1.0 / sr)
    mag    = np.abs(Y) / N * 2
    mag_db = 20 * np.log10(mag + 1e-12)
    return freqs, mag, mag_db

def zone_energy(freqs, mag, zone_hz):
    lo, hi = zone_hz
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sum(mag[mask] ** 2))

def add_zone_spans(ax):
    for (name, (lo, hi)), zc in zip(FREQ_ZONES.items(), ZONE_COLORS):
        ax.axvspan(lo, hi, alpha=0.07, color=zc)

def style_ax(ax):
    ax.set_xscale("log")
    ax.set_xlim(20, SR // 2)
    ax.grid(True, alpha=0.2, linestyle="--", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)



# Graf 1 - FFT spektar jednog fajla
def plot_fft(y, freqs, mag_db, fname):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"FFT spektar — {fname}", fontsize=13, fontweight="bold")

    # gornji graf nam prikazuje cijeli spekter
    ax = axes[0]
    ax.plot(freqs, mag_db, color="#3498db", linewidth=0.9, alpha=0.9)
    ax.fill_between(freqs, mag_db, mag_db.min(), alpha=0.15, color="#3498db")
    add_zone_spans(ax)

    legend_els = [Patch(facecolor=zc, alpha=0.4, label=name.replace("\n", " ")) for (name, _), zc in zip(FREQ_ZONES.items(), ZONE_COLORS)]
    ax.legend(handles=legend_els, fontsize=7, ncol=3, loc="lower left", framealpha=0.85)
    ax.set_ylabel("Magnituda (dB)", fontsize=10)
    ax.set_title("Cijeli spektar (20 Hz - Nyquist)", fontsize=10)
    style_ax(ax)

    # donji graf ima zoom na 20–2000 Hz (gdje je za BIC najvažnije)
    ax2 = axes[1]
    mask = freqs <= 2000
    ax2.plot(freqs[mask], mag_db[mask], color="#e74c3c", linewidth=1.2)
    ax2.fill_between(freqs[mask], mag_db[mask], mag_db[mask].min(),
                     alpha=0.15, color="#e74c3c")
    add_zone_spans(ax2)

    # oznacimo vrh koja je dominantna frek
    peak_idx = np.argmax(mag_db[mask])
    peak_f   = freqs[mask][peak_idx]
    peak_db  = mag_db[mask][peak_idx]
    ax2.annotate(f"  peak: {peak_f:.1f} Hz\n  {peak_db:.1f} dB", xy=(peak_f, peak_db), xytext=(peak_f * 2, peak_db - 5), arrowprops=dict(arrowstyle="->", color="black", lw=1),fontsize=8, fontweight="bold")

    ax2.set_xlabel("Frekvencija (Hz)", fontsize=10)
    ax2.set_ylabel("Magnituda (dB)", fontsize=10)
    ax2.set_title("Zoom: 20 – 2000 Hz", fontsize=10)
    style_ax(ax2)

    plt.tight_layout()
    out = OUTPUT_DIR / "01_fft.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  --> {out}")



# Graf 2 — STFT spektrogram
def plot_stft(y, fname):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"STFT spektrogram — {fname}", fontsize=12, fontweight="bold")

    # Lijevo stoji log y-os (mel-like prikaz)
    D    = librosa.stft(y, n_fft=2048, hop_length=512)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    img1 = librosa.display.specshow(D_db, sr=SR, hop_length=512, _axis="time", y_axis="log", ax=axes[0], cmap="magma")
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB", pad=0.02)
    axes[0].set_title("Log frekvencijska os", fontsize=10)
    axes[0].set_xlabel("Vrijeme (s)")
    axes[0].set_ylabel("Frekvencija (Hz)")

    # Desno stojilinearna y-os, zoom 0–3kHz
    img2 = librosa.display.specshow(D_db, sr=SR, hop_length=512, x_axis="time", y_axis="linear", ax=axes[1], cmap="magma")
    axes[1].set_ylim(0, 3000)
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB", pad=0.02)
    axes[1].set_title("Linearna os — zoom 0–3 kHz", fontsize=10)
    axes[1].set_xlabel("Vrijeme (s)")
    axes[1].set_ylabel("Frekvencija (Hz)")

    plt.tight_layout()
    out = OUTPUT_DIR / "02_stft_spektrogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  --> {out}")



# Graf 3 - Energija po frekvencijskim zonama
def plot_zone_energy(freqs, mag, fname):
    zone_names = list(FREQ_ZONES.keys())
    energies   = [zone_energy(freqs, mag, FREQ_ZONES[z]) for z in zone_names]
    total      = sum(energies)
    pcts       = [e / total * 100 for e in energies]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Distribucija energije po frekvencijskim zonama — {fname}",
                 fontsize=12, fontweight="bold")

    # bar chart
    bars = axes[0].bar(range(len(zone_names)), pcts, color=ZONE_COLORS, alpha=0.85, edgecolor="white", width=0.6)
    for bar, pct in zip(bars, pcts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9,fontweight="bold")
    axes[0].set_xticks(range(len(zone_names)))
    axes[0].set_xticklabels([z.replace("\n", " ") for z in zone_names], fontsize=8, rotation=15, ha="right")
    axes[0].set_ylabel("% ukupne energije", fontsize=10)
    axes[0].set_title("Udio energije po zoni", fontsize=11)
    axes[0].grid(True, alpha=0.2, axis="y", linestyle="--")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # pie chart
    axes[1].pie(pcts, labels=[z.replace("\n", " ") for z in zone_names],
                colors=ZONE_COLORS, autopct="%1.1f%%", startangle=90,
                pctdistance=0.78, textprops={"fontsize": 8})
    axes[1].set_title("Udio energije — pie", fontsize=11)
    plt.tight_layout()
    out = OUTPUT_DIR / "03_frekvencijske_zone.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  --> {out}")

    # ispis
    print("\n  Frekvencijska distribucija energije:")
    dominant = zone_names[np.argmax(pcts)]
    print(f"  Dominantna zona: {dominant.replace(chr(10), ' ')} ({max(pcts):.1f}%)")
    for z, p in zip(zone_names, pcts):
        bar_vis = "█" * int(p / 2)
        print(f"  {z.replace(chr(10), ' '):22s}: {p:5.1f}%  {bar_vis}")


# Graf 4 — Augmentacije preview
def plot_augmentations(y_orig, freqs, mag_db_orig, fname):

    def aug_time_stretch(y, rate):
        return librosa.effects.time_stretch(y, rate=rate)

    def aug_pitch_shift(y, steps):
        return librosa.effects.pitch_shift(y, sr=SR, n_steps=steps)

    def aug_noise(y, snr_db):
        sig_p   = np.mean(y ** 2)
        noise_p = sig_p / (10 ** (snr_db / 10))
        return y + np.random.normal(0, np.sqrt(noise_p), len(y))

    def aug_lowpass(y, cutoff):
        b, a = butter(4, cutoff / (SR / 2), btype="low")
        return filtfilt(b, a, y)

    def aug_highpass(y, cutoff):
        b, a = butter(4, cutoff / (SR / 2), btype="high")
        return filtfilt(b, a, y)

    def aug_gain(y, db):
        return y * (10 ** (db / 20))

    augmentations = {
        "Original":               y_orig,
        "Time stretch (×1.3)":    aug_time_stretch(y_orig, 1.3),
        "Time stretch (×0.7)":    aug_time_stretch(y_orig, 0.7),
        "Pitch shift (+2 st)":    aug_pitch_shift(y_orig, 2),
        "Pitch shift (−2 st)":    aug_pitch_shift(y_orig, -2),
        "Bijeli šum (SNR 20dB)":  aug_noise(y_orig, 20),
        "Bijeli šum (SNR 10dB)":  aug_noise(y_orig, 10),
        "Low-pass (4kHz)":        aug_lowpass(y_orig, 4000),
        "High-pass (150Hz)":      aug_highpass(y_orig, 150),
        "Gain +6dB":              aug_gain(y_orig, 6),
    }

    target = int(SR * DURATION)
    n      = len(augmentations)
    cols   = 2
    rows   = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
    fig.suptitle(f"Utjecaj augmentacija na FFT spektar — {fname}",fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for idx, (aug_name, y_aug) in enumerate(augmentations.items()):
        ax      = axes[idx]
        is_orig = aug_name == "Original"

        # Pad/crop
        if len(y_aug) < target:
            y_aug = np.pad(y_aug, (0, target - len(y_aug)))
        else:
            y_aug = y_aug[:target]

        _, _, mag_db_aug = compute_fft(y_aug)

        if is_orig:
            ax.plot(freqs, mag_db_orig, color="#e74c3c", linewidth=2.0,label="Original")
            ax.fill_between(freqs, mag_db_orig, mag_db_orig.min(), alpha=0.12, color="#e74c3c")
        else:
            ax.plot(freqs, mag_db_orig, color="#e74c3c", linewidth=1.0, alpha=0.4, linestyle="--", label="Original")
            ax.plot(freqs, mag_db_aug, color="#3498db", linewidth=1.4, label=aug_name)

            # razlika decibela na sekundarnoj osi
            diff = mag_db_aug - mag_db_orig
            ax2  = ax.twinx()
            ax2.plot(freqs, diff, color="#2ecc71", linewidth=0.8,
                     alpha=0.55, linestyle=":")
            ax2.axhline(0, color="#2ecc71", linewidth=0.5, alpha=0.3)
            ax2.set_ylabel("Δ dB", fontsize=7, color="#2ecc71")
            ax2.tick_params(labelsize=6, colors="#2ecc71")
            ax2.set_ylim(-35, 35)
            ax2.spines["top"].set_visible(False)

        style_ax(ax)
        ax.set_xlabel("Frekvencija (Hz)", fontsize=8)
        ax.set_ylabel("Magnituda (dB)", fontsize=8)
        ax.set_title(aug_name, fontsize=9, fontweight="bold",
                     color="#e74c3c" if is_orig else "#2c3e50")
        ax.legend(fontsize=7, loc="lower left")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "04_augmentacija_preview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {out}")



def main():
    fpath = Path(FILE_PATH)
    if not fpath.exists():
        print(f"File nije pronađen: {FILE_PATH}")
        return

    fname = fpath.name
    print("=" * 60)
    print(f"  FOURIER ANALIZA — {fname}")
    print("=" * 60)
    print(f"  Putanja: {fpath.resolve()}")

    print("\n  Učitavam audio...")
    y = load_audio()
    duration_real = librosa.get_duration(y=y, sr=SR)
    rms = float(np.sqrt(np.mean(y ** 2)))
    print(f"  Trajanje: {duration_real:.2f}s | RMS: {rms:.5f} | SR: {SR} Hz")

    print("\n  Računam FFT...")
    freqs, mag, mag_db = compute_fft(y)

    print("\n[1/4] FFT spektar")
    plot_fft(y, freqs, mag_db, fname)

    print("[2/4] STFT spektrogram")
    plot_stft(y, fname)

    print("[3/4] Energija po zonama")
    plot_zone_energy(freqs, mag, fname)

    print("[4/4] Augmentacije preview")
    plot_augmentations(y, freqs, mag_db, fname)

    print(f"""
{"="*60}
  GOTOVO! --> {OUTPUT_DIR}/
{"="*60}
  01_fft.png
    --> Cijeli spektar + zoom 20-2000 Hz s peak oznakom

  02_stft_spektrogram.png
    --> Log os + linearna os zoom 0-3kHz
       (vidiš kako se frekvencije mijenjaju kroz VRIJEME)

  03_frekvencijske_zone.png
     --> Gdje je energija — bar + pie

  04_augmentacija_preview.png
     --> Zelena linija = Δ dB razlika od originala
       Ravnija zelena = sigurnija augmentacija
    """)


if __name__ == "__main__":
    main()