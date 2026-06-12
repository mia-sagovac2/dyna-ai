import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
import pandas as pd


FILE_A = "../fourier_separated_audio/percussive.wav"
FILE_B = "../data/all_sorted/BIC/5ceeff0b-2023-08-04_10-25-40.0-Swanston6_TJN_NoV_L1_1Lane_Idling_mono.wav"   # konkretni solo file kojeg zelimo usporediti

LABEL_A = "Čisti bicikl" 
LABEL_B = "Bicikl + šum"


OUTPUT_DIR = Path("../eda_analiza_specific_audio_comparison")
SR_TARGET  = 48000
N_MFCC     = 13
DURATION   = None

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_A = "#3498db"
COLOR_B = "#e74c3c"
COLOR_D = "#2ecc71"

FREQ_ZONES = {
    "Sub-bass\n(20-60)":   (20,    60),
    "Bass\n(60-250)":      (60,   250),
    "Low-mid\n(250-500)":  (250,  500),
    "Mid\n(500-2k)":       (500,  2000),
    "High-mid\n(2k-6k)":  (2000, 6000),
    "High\n(6k-20k)":     (6000, 20000),
}
ZONE_COLORS = ["#2c3e50","#2980b9","#27ae60","#f39c12","#e74c3c","#9b59b6"]


# loading
def load_both():
    results = {}
    for label, path in [(LABEL_A, FILE_A), (LABEL_B, FILE_B)]:
        fpath = Path(path)
        if not fpath.exists():
            raise FileNotFoundError(f"Fajl nije pronađen: {path}")
        y, sr = librosa.load(str(fpath), sr=SR_TARGET, mono=True, duration=DURATION)
        results[label] = {"y": y, "sr": sr, "name": fpath.name, "path": path}
        dur = librosa.get_duration(y=y, sr=sr)
        rms = float(np.sqrt(np.mean(y**2)))
        print(f"  [{label}]  {fpath.name}")
        print(f"    Trajanje: {dur:.3f}s | RMS: {rms:.6f} | SR: {sr}Hz")

    # izjednacavanje duljina (ukoliko potrebno)
    ya, yb = results[LABEL_A]["y"], results[LABEL_B]["y"]
    min_len = min(len(ya), len(yb))
    results[LABEL_A]["y_trim"] = ya[:min_len]
    results[LABEL_B]["y_trim"] = yb[:min_len]
    results["min_len"] = min_len
    results["sr"] = SR_TARGET
    return results


# helperi
def compute_fft(y, sr):
    N      = len(y)
    win    = np.hanning(N)
    Y      = np.fft.rfft(y * win)
    freqs  = np.fft.rfftfreq(N, d=1.0 / sr)
    mag    = np.abs(Y) / N * 2
    mag_db = 20 * np.log10(mag + 1e-12)
    return freqs, mag, mag_db


def extract_frame_features(y, sr, hop=512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=hop)
    return {
        "RMS":               librosa.feature.rms(y=y, hop_length=hop)[0],
        "ZCR":               librosa.feature.zero_crossing_rate(y, hop_length=hop)[0],
        "Spectral centroid": librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0],
        "Spectral bandwidth":librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0],
        "Spectral rolloff":  librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0],
        "Spectral flatness": librosa.feature.spectral_flatness(y=y, hop_length=hop)[0],
        "Onset strength":    librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop),
        "MFCC 1":            mfcc[0],
        "MFCC 2":            mfcc[1],
        "MFCC 3":            mfcc[2],
    }


def extract_summary_features(y, sr):
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr       = librosa.feature.zero_crossing_rate(y)
    rms       = librosa.feature.rms(y=y)
    flatness  = librosa.feature.spectral_flatness(y=y)
    contrast  = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    chroma    = librosa.feature.chroma_stft(y=y, sr=sr)
    return {
        "MFCC mean":          float(np.mean(mfcc)),
        "MFCC std":           float(np.std(mfcc)),
        "Spectral centroid":  float(np.mean(centroid)),
        "Spectral bandwidth": float(np.mean(bandwidth)),
        "Spectral rolloff":   float(np.mean(rolloff)),
        "ZCR mean":           float(np.mean(zcr)),
        "RMS mean":           float(np.mean(rms)),
        "Spectral flatness":  float(np.mean(flatness)),
        "Spectral contrast":  float(np.mean(contrast)),
        "Onset strength":     float(np.mean(onset_env)),
        "Onset std":          float(np.std(onset_env)),
        "Chroma mean":        float(np.mean(chroma)),
    }


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle="--")



# Graf 1 — Waveform + RMS + ZCR usporedba
def plot_waveform(data):
    sr = data["sr"]
    hop = 512
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex="col")
    fig.suptitle("Waveform, RMS i ZCR — usporedba", fontsize=13, fontweight="bold")

    for col, (label, color) in enumerate([(LABEL_A, COLOR_A), (LABEL_B, COLOR_B)]):
        y = data[label]["y"]
        rms   = librosa.feature.rms(y=y, hop_length=hop)[0]
        zcr   = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
        t_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
        t_zcr = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=hop)

        librosa.display.waveshow(y, sr=sr, ax=axes[0, col], color=color, alpha=0.8)
        axes[0, col].set_title(f"{label} — waveform", fontsize=10, fontweight="bold")
        axes[0, col].set_ylabel("Amplituda")
        style(axes[0, col])

        axes[1, col].plot(t_rms, rms, color=color, linewidth=1.2)
        axes[1, col].fill_between(t_rms, rms, alpha=0.2, color=color)
        axes[1, col].axhline(np.mean(rms), color="black", linestyle="--",
                              linewidth=1, label=f"mean: {np.mean(rms):.5f}")
        axes[1, col].set_title(f"{label} — RMS", fontsize=10)
        axes[1, col].set_ylabel("RMS")
        axes[1, col].legend(fontsize=7)
        style(axes[1, col])

        axes[2, col].plot(t_zcr, zcr, color=color, linewidth=1.0, alpha=0.85)
        axes[2, col].fill_between(t_zcr, zcr, alpha=0.15, color=color)
        axes[2, col].axhline(np.mean(zcr), color="black", linestyle="--",
                              linewidth=1, label=f"mean: {np.mean(zcr):.4f}")
        axes[2, col].set_title(f"{label} — ZCR", fontsize=10)
        axes[2, col].set_ylabel("ZCR")
        axes[2, col].set_xlabel("Vrijeme (s)")
        axes[2, col].legend(fontsize=7)
        style(axes[2, col])

    plt.tight_layout()
    out = OUTPUT_DIR / "01_waveform_usporedba.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {out}")


# Graf 2 — Mel-spektrogrami i MFCC side-by-side
def plot_spektrogram(data):
    sr = data["sr"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("Mel-spektrogram i MFCC — usporedba", fontsize=13, fontweight="bold")

    for col, (label, color) in enumerate([(LABEL_A, COLOR_A), (LABEL_B, COLOR_B)]):
        y = data[label]["y"]

        mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img1   = librosa.display.specshow(mel_db, sr=sr, x_axis="time",
                                          y_axis="mel", ax=axes[0, col], cmap="magma")
        fig.colorbar(img1, ax=axes[0, col], format="%+2.0f dB", pad=0.02)
        axes[0, col].set_title(f"{label} — mel-spektrogram", fontsize=10, fontweight="bold")
        axes[0, col].set_xlabel("Vrijeme (s)")
        axes[0, col].set_ylabel("Frekvencija (mel)")

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        img2 = librosa.display.specshow(mfcc, sr=sr, x_axis="time",
                                        ax=axes[1, col], cmap="coolwarm")
        fig.colorbar(img2, ax=axes[1, col], pad=0.02)
        axes[1, col].set_yticks(range(N_MFCC))
        axes[1, col].set_yticklabels([f"C{i+1}" for i in range(N_MFCC)], fontsize=7)
        axes[1, col].set_title(f"{label} — MFCC", fontsize=10, fontweight="bold")
        axes[1, col].set_xlabel("Vrijeme (s)")

    plt.tight_layout()
    out = OUTPUT_DIR / "02_spektrogram_usporedba.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {out}")


# Graf 3 — FFT usporedba na istom grafu + razlika
def plot_fft(data):
    sr = data["sr"]
    ya = data[LABEL_A]["y_trim"]
    yb = data[LABEL_B]["y_trim"]

    freqs, mag_a, mag_db_a = compute_fft(ya, sr)
    _,     mag_b, mag_db_b = compute_fft(yb, sr)
    diff_db = mag_db_b - mag_db_a   # pozitivno = B ima više energije

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("FFT spektar — usporedba", fontsize=13, fontweight="bold")

    # na gornjem su oba spektra skupa
    axes[0].plot(freqs, mag_db_a, color=COLOR_A, linewidth=1.2,
                 label=LABEL_A, alpha=0.9)
    axes[0].plot(freqs, mag_db_b, color=COLOR_B, linewidth=1.2,
                 label=LABEL_B, alpha=0.9)
    axes[0].set_title("Oba spektra — overlay", fontsize=11)
    axes[0].legend(fontsize=9)

    # srednji je vizalni prikaz razlike
    axes[1].fill_between(freqs, mag_db_a, mag_db_b,
                         where=(mag_db_b >= mag_db_a),
                         alpha=0.5, color=COLOR_B,
                         label=f"{LABEL_B} > {LABEL_A}")
    axes[1].fill_between(freqs, mag_db_a, mag_db_b,
                         where=(mag_db_b < mag_db_a),
                         alpha=0.5, color=COLOR_A,
                         label=f"{LABEL_A} > {LABEL_B}")
    axes[1].plot(freqs, mag_db_a, color=COLOR_A, linewidth=0.8, alpha=0.6)
    axes[1].plot(freqs, mag_db_b, color=COLOR_B, linewidth=0.8, alpha=0.6)
    axes[1].set_title("Razlika — obojano gdje koji fajl ima više energije", fontsize=11)
    axes[1].legend(fontsize=8)

    # donji razlika u decibelima
    axes[2].plot(freqs, diff_db, color=COLOR_D, linewidth=1.0, alpha=0.85)
    axes[2].fill_between(freqs, diff_db, 0,
                         where=(diff_db > 0), alpha=0.3, color=COLOR_B,
                         label=f"{LABEL_B} viši")
    axes[2].fill_between(freqs, diff_db, 0,
                         where=(diff_db < 0), alpha=0.3, color=COLOR_A,
                         label=f"{LABEL_A} viši")
    axes[2].axhline(0, color="black", linewidth=1.0, linestyle="--")
    axes[2].set_title("Δ dB (B − A): pozitivno = B ima više energije", fontsize=11)
    axes[2].set_ylabel("Δ dB")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(20, sr // 2)
        ax.set_ylabel("Magnituda (dB)")
        ax.set_xlabel("Frekvencija (Hz)")
        for (name, (lo, hi)), zc in zip(FREQ_ZONES.items(), ZONE_COLORS):
            ax.axvspan(lo, hi, alpha=0.05, color=zc)
        style(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "03_fft_usporedba.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {out}")


# Graf 4 — Feature summary usporedba (bar + radar)
def plot_features_summary(data):
    sr  = data["sr"]
    fa  = extract_summary_features(data[LABEL_A]["y"], sr)
    fb  = extract_summary_features(data[LABEL_B]["y"], sr)
    names = list(fa.keys())

    # relative diff (%) = (B - A) / |A| * 100
    rel_diff = {}
    for n in names:
        a, b = fa[n], fb[n]
        rel_diff[n] = (b - a) / (abs(a) + 1e-12) * 100

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Feature summary — usporedba", fontsize=13, fontweight="bold")

    # bar oba filea po feateru
    all_vals = [abs(fa[n]) for n in names] + [abs(fb[n]) for n in names]
    max_v    = max(all_vals) if max(all_vals) > 0 else 1
    norm_a   = [abs(fa[n]) / max_v for n in names]
    norm_b   = [abs(fb[n]) / max_v for n in names]

    y_pos = np.arange(len(names))
    bar_h = 0.35
    bars_a = axes[0].barh(y_pos + bar_h/2, norm_a, bar_h,
                           color=COLOR_A, alpha=0.8, label=LABEL_A)
    bars_b = axes[0].barh(y_pos - bar_h/2, norm_b, bar_h,
                           color=COLOR_B, alpha=0.8, label=LABEL_B)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_title("Normalizirane vrijednosti featura", fontsize=10)
    axes[0].set_xlabel("Normalizirana vrijednost")
    axes[0].legend(fontsize=8)
    style(axes[0])

    # bar kao relativna razlika
    diffs  = [rel_diff[n] for n in names]
    colors = [COLOR_B if d > 0 else COLOR_A for d in diffs]
    axes[1].barh(names, diffs, color=colors, alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=1.0, linestyle="--")
    for i, (n, d) in enumerate(zip(names, diffs)):
        axes[1].text(d + (1 if d >= 0 else -1), i,
                     f"{d:+.1f}%", va="center", fontsize=7,
                     color=COLOR_B if d > 0 else COLOR_A)
    axes[1].set_title(f"Relativna razlika %\n({LABEL_B} − {LABEL_A}) / |{LABEL_A}|",
                      fontsize=10)
    axes[1].set_xlabel("Razlika (%)")
    style(axes[1])

    # ispis u konzoli
    print("\n  Relativna odstupanja featura (|razlika| > 20%):")
    printed = False
    for n, d in sorted(rel_diff.items(), key=lambda x: -abs(x[1])):
        if abs(d) > 20:
            marker = "⚠" if abs(d) > 50 else "~"
            print(f"  {marker} {n:25s}: {d:+.1f}%")
            printed = True
    if not printed:
        print("  Nema velikih odstupanja (svi featurei unutar ±20%)")

    # radar na oba filea
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    angles += angles[:1]

    ax_r = axes[2]
    ax_r.remove()
    ax_r = fig.add_subplot(1, 3, 3, polar=True)

    for vals, color, label in [(norm_a, COLOR_A, LABEL_A),
                                (norm_b, COLOR_B, LABEL_B)]:
        v = vals + vals[:1]
        ax_r.plot(angles, v, color=color, linewidth=2, label=label)
        ax_r.fill(angles, v, color=color, alpha=0.15)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(names, fontsize=6.5)
    ax_r.set_title("Radar — profil featura", fontsize=10, pad=15)
    ax_r.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax_r.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "04_features_usporedba.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {out}")


# Graf 5 — MFCC mean/std usporedba
def plot_mfcc(data):
    sr   = data["sr"]
    coefs = np.arange(1, N_MFCC + 1)

    mfcc_a = librosa.feature.mfcc(y=data[LABEL_A]["y"], sr=sr, n_mfcc=N_MFCC)
    mfcc_b = librosa.feature.mfcc(y=data[LABEL_B]["y"], sr=sr, n_mfcc=N_MFCC)

    means_a, stds_a = np.mean(mfcc_a, axis=1), np.std(mfcc_a, axis=1)
    means_b, stds_b = np.mean(mfcc_b, axis=1), np.std(mfcc_b, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("MFCC detaljna usporedba", fontsize=13, fontweight="bold")

    # Mean usporedba
    w = 0.35
    axes[0, 0].bar(coefs - w/2, means_a, w, color=COLOR_A, alpha=0.8, label=LABEL_A)
    axes[0, 0].bar(coefs + w/2, means_b, w, color=COLOR_B, alpha=0.8, label=LABEL_B)
    axes[0, 0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 0].set_title("MFCC mean po koeficijentu", fontsize=10)
    axes[0, 0].set_xlabel("MFCC koeficijent")
    axes[0, 0].set_ylabel("Mean vrijednost")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_xticks(coefs)
    style(axes[0, 0])

    # Std usporedba
    axes[0, 1].bar(coefs - w/2, stds_a, w, color=COLOR_A, alpha=0.8, label=LABEL_A)
    axes[0, 1].bar(coefs + w/2, stds_b, w, color=COLOR_B, alpha=0.8, label=LABEL_B)
    axes[0, 1].set_title("MFCC std po koeficijentu", fontsize=10)
    axes[0, 1].set_xlabel("MFCC koeficijent")
    axes[0, 1].set_ylabel("Std vrijednost")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_xticks(coefs)
    style(axes[0, 1])

    # Mean ± std — error bar oba
    axes[1, 0].errorbar(coefs - 0.1, means_a, yerr=stds_a, fmt="o-",
                        color=COLOR_A, elinewidth=1.2, capsize=4,
                        linewidth=1.8, markersize=5, label=LABEL_A)
    axes[1, 0].errorbar(coefs + 0.1, means_b, yerr=stds_b, fmt="s-",
                        color=COLOR_B, elinewidth=1.2, capsize=4,
                        linewidth=1.8, markersize=5, label=LABEL_B)
    axes[1, 0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 0].set_title("MFCC mean ± std (error bar)", fontsize=10)
    axes[1, 0].set_xlabel("MFCC koeficijent")
    axes[1, 0].set_ylabel("Vrijednost")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_xticks(coefs)
    style(axes[1, 0])

    # Razlika meana po koeficijentu
    diff_means = means_b - means_a
    bar_colors = [COLOR_B if d > 0 else COLOR_A for d in diff_means]
    axes[1, 1].bar(coefs, diff_means, color=bar_colors, alpha=0.8, edgecolor="white")
    axes[1, 1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 1].set_title(f"MFCC razlika meana (B − A)\n{LABEL_B} − {LABEL_A}",
                         fontsize=10)
    axes[1, 1].set_xlabel("MFCC koeficijent")
    axes[1, 1].set_ylabel("Δ mean")
    axes[1, 1].set_xticks(coefs)
    style(axes[1, 1])

    plt.tight_layout()
    out = OUTPUT_DIR / "05_mfcc_usporedba.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {out}")



# Graf 6 — PDF i CDF svakog featura — oba fajla na istom grafu
def plot_pdf_cdf(data):
    sr  = data["sr"]
    ff_a = extract_frame_features(data[LABEL_A]["y"], sr)
    ff_b = extract_frame_features(data[LABEL_B]["y"], sr)
    feat_names = list(ff_a.keys())

    rows = len(feat_names)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
    fig.suptitle(f"PDF i CDF featura po frejmovima — usporedba",
                 fontsize=13, fontweight="bold", y=1.005)

    for row, feat in enumerate(feat_names):
        ax_pdf = axes[row][0]
        ax_cdf = axes[row][1]

        for vals, color, label in [
            (ff_a[feat], COLOR_A, LABEL_A),
            (ff_b[feat], COLOR_B, LABEL_B),
        ]:
            vals = vals[np.isfinite(vals)]
            if len(vals) < 5:
                continue

            x_min, x_max = vals.min(), vals.max()
            margin = (x_max - x_min) * 0.1 + 1e-12

            # PDF
            if np.std(vals) < 1e-10 or len(np.unique(vals)) < 3:
                ax_pdf.axvline(vals[0], color=color, linewidth=2,
                               linestyle="--", label=label)
            else:
                try:
                    kde    = gaussian_kde(vals, bw_method="scott")
                    x_grid = np.linspace(x_min - margin, x_max + margin, 400)
                    y_kde  = kde(x_grid)
                    ax_pdf.plot(x_grid, y_kde, color=color, linewidth=2, label=label)
                    ax_pdf.fill_between(x_grid, y_kde, alpha=0.15, color=color)
                except Exception:
                    ax_pdf.hist(vals, bins=40, density=True, color=color,
                                alpha=0.45, label=label, histtype="stepfilled")

            # CDF
            sv = np.sort(vals)
            cy = np.arange(1, len(sv) + 1) / len(sv)
            ax_cdf.plot(sv, cy, color=color, linewidth=2, label=label)

            # KS statistika (koliko su distribucije različite)
            # Samo jednom (kad je oba dostupna)

        ax_pdf.set_title(f"PDF — {feat}", fontsize=10, fontweight="bold")
        ax_pdf.set_xlabel(feat, fontsize=9)
        ax_pdf.set_ylabel("Gustoća", fontsize=9)
        ax_pdf.legend(fontsize=8, framealpha=0.85)
        ax_pdf.grid(True, alpha=0.2, linestyle="--")
        ax_pdf.spines["top"].set_visible(False)
        ax_pdf.spines["right"].set_visible(False)

        ax_cdf.set_title(f"CDF — {feat}", fontsize=10, fontweight="bold")
        ax_cdf.set_xlabel(feat, fontsize=9)
        ax_cdf.set_ylabel("P(X ≤ x)", fontsize=9)
        ax_cdf.set_ylim(0, 1.05)
        ax_cdf.legend(fontsize=8, framealpha=0.85)
        ax_cdf.grid(True, alpha=0.2, linestyle="--")
        ax_cdf.spines["top"].set_visible(False)
        ax_cdf.spines["right"].set_visible(False)
        for q in [0.25, 0.50, 0.75]:
            ax_cdf.axhline(q, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # KS test — konzolni ispis
    from scipy.stats import ks_2samp
    print("\n  KS test (koliko se distribucije razlikuju, p < 0.05 = statistički različite):")
    for feat in feat_names:
        va = ff_a[feat]
        vb = ff_b[feat]
        va = va[np.isfinite(va)]
        vb = vb[np.isfinite(vb)]
        if len(va) > 5 and len(vb) > 5:
            stat, p = ks_2samp(va, vb)
            sig = "⚠ RAZLIČITE" if p < 0.05 else "  slične"
            print(f"  {sig}  {feat:25s}: KS={stat:.3f}, p={p:.4f}")

    plt.tight_layout()
    out = OUTPUT_DIR / "06_pdf_cdf_usporedba.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {out}")


# Graf 7 — Energija po frekvencijskim zonama
def plot_zone_energy(data):
    sr = data["sr"]
    zone_names = list(FREQ_ZONES.keys())
    results = {}

    for label in [LABEL_A, LABEL_B]:
        y = data[label]["y_trim"]
        freqs, mag, _ = compute_fft(y, sr)
        energies = []
        for name, (lo, hi) in FREQ_ZONES.items():
            mask = (freqs >= lo) & (freqs < hi)
            energies.append(float(np.sum(mag[mask] ** 2)))
        total = sum(energies)
        results[label] = [e / total * 100 for e in energies]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Energija po frekvencijskim zonama — usporedba", fontsize=13, fontweight="bold")

    x     = np.arange(len(zone_names))
    w     = 0.35
    short = [z.replace("\n", " ") for z in zone_names]

    # Grouped bar
    axes[0].bar(x - w/2, results[LABEL_A], w, color=COLOR_A,
                alpha=0.85, label=LABEL_A, edgecolor="white")
    axes[0].bar(x + w/2, results[LABEL_B], w, color=COLOR_B,
                alpha=0.85, label=LABEL_B, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short, fontsize=8, rotation=15, ha="right")
    axes[0].set_ylabel("% ukupne energije")
    axes[0].set_title("Udio energije po zoni", fontsize=11)
    axes[0].legend(fontsize=9)
    style(axes[0])

    # Razlika zona
    diff_zones = [results[LABEL_B][i] - results[LABEL_A][i]
                  for i in range(len(zone_names))]
    bar_cols = [COLOR_B if d > 0 else COLOR_A for d in diff_zones]
    axes[1].bar(x, diff_zones, color=bar_cols, alpha=0.85, edgecolor="white")
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short, fontsize=8, rotation=15, ha="right")
    axes[1].set_ylabel("Δ % energije (B − A)")
    axes[1].set_title(f"Razlika po zoni\n{LABEL_B} − {LABEL_A}", fontsize=11)
    style(axes[1])

    out = OUTPUT_DIR / "07_frekvencijske_zone.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {out}")



def main():
    print("=" * 60)
    print("  EDA — USPOREDBA DVA AUDIO FAJLA")
    print(f"  A: {LABEL_A}")
    print(f"  B: {LABEL_B}")
    print("=" * 60)

    data = load_both()

    print("\n[1/7] Waveform + RMS + ZCR")
    plot_waveform(data)

    print("[2/7] Mel-spektrogram + MFCC")
    plot_spektrogram(data)

    print("[3/7] FFT usporedba")
    plot_fft(data)

    print("[4/7] Feature summary + radar")
    plot_features_summary(data)

    print("[5/7] MFCC detalji")
    plot_mfcc(data)

    print("[6/7] PDF i CDF + KS test")
    plot_pdf_cdf(data)

    print("[7/7] Energija po frekvencijskim zonama")
    plot_zone_energy(data)

    print(f"""
{"="*60}
  GOTOVO! --> {OUTPUT_DIR}/
{"="*60}
  01_waveform_usporedba.png   — waveform/RMS/ZCR side-by-side
  02_spektrogram_usporedba.png — mel + MFCC heatmap side-by-side
  03_fft_usporedba.png        — overlay + fill_between + Δ dB
  04_features_usporedba.png   — bar (grouped) + % razlika + radar
  05_mfcc_usporedba.png       — mean/std/errorbar/Δ mean po koef.
  06_pdf_cdf_usporedba.png    — PDF+CDF na istom grafu + KS test
  07_frekvencijske_zone.png   — grouped bar + Δ zona + sažetak
    """)


if __name__ == "__main__":
    main()