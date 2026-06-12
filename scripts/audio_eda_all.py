import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict
import warnings
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore")

DATASET_DIR = "../data/all_sorted/"
OUTPUT_DIR  = "../eda_analiza"
SR_TARGET   = 48000
N_MFCC      = 13
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/pdf_cdf", exist_ok=True)

PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6","#1abc9c", "#e67e22", "#e91e63", "#00bcd4", "#8bc34a"]

def class_color(i):
    return PALETTE[i % len(PALETTE)]


# helper: PDF + CDF za jednu varijablu, sve klase na istom grafu
def plot_pdf_cdf(data_per_class: dict, feature_name: str, xlabel: str, filename: str, log_scale: bool = False):
    klase = list(data_per_class.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"PDF i CDF - {feature_name}", fontsize=13, fontweight="bold")

    for i, klasa in enumerate(klase):
        vals = np.array([v for v in data_per_class[klasa] if np.isfinite(v)])
        if len(vals) < 5:
            continue
        color = class_color(i)

        if log_scale:
            vals = vals[vals > 0]
            plot_vals = np.log10(vals)
        else:
            plot_vals = vals

        # histogram za PDF
        x_min, x_max = plot_vals.min(), plot_vals.max()
        margin = (x_max - x_min) * 0.1 + 1e-9

        if np.std(plot_vals) < 1e-10 or len(np.unique(plot_vals)) < 3:
            # Sve vrijednosti iste → prikaži kao vertikalnu liniju
            axes[0].axvline(plot_vals[0], color=color, linewidth=2, linestyle="--", label=f"{klasa} (konst. {plot_vals[0]:.4g})")
        else:
            try:
                kde    = gaussian_kde(plot_vals, bw_method="scott")
                x_grid = np.linspace(x_min - margin, x_max + margin, 400)
                y_kde  = kde(x_grid)
                axes[0].plot(x_grid, y_kde, color=color, linewidth=2, label=klasa)
                axes[0].fill_between(x_grid, y_kde, alpha=0.12, color=color)
            except Exception:
                # normalizirani histogram kao fallback
                axes[0].hist(plot_vals, bins=30, density=True, color=color, alpha=0.45, label=f"{klasa} (hist)", histtype="stepfilled")
                axes[0].hist(plot_vals, bins=30, density=True, color=color, alpha=0.9, histtype="step", linewidth=1.5)

        # CDF
        sorted_vals = np.sort(plot_vals)
        cdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        axes[1].plot(sorted_vals, cdf_y, color=color, linewidth=2, label=klasa)

    for ax, title in zip(axes, ["PDF (gustoća vjerojatnosti)", "CDF (kumulativna distribucija)"]):
        ax.set_title(title, fontsize=11)
        if log_scale:
            ax.set_xlabel(f"log₁₀({xlabel})", fontsize=10)
        else:
            ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Gustoća" if ax == axes[0] else "Vjerojatnost", fontsize=10)
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.25, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    axes[1].axhline(0.50, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    axes[1].axhline(0.75, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    axes[1].text(axes[1].get_xlim()[0], 0.51, " Q2 (medijan)", fontsize=7, color="gray")
    axes[1].text(axes[1].get_xlim()[0], 0.26, " Q1", fontsize=7, color="gray")
    axes[1].text(axes[1].get_xlim()[0], 0.76, " Q3", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {filename}")


def plot_pdf_cdf_grid(df_feat: pd.DataFrame, feature_cols: list,
                      klase: list, output_path: str):
    """
    Veliki grid: svaki feature --> jedan red (PDF lijevo, CDF desno)
    """
    n = len(feature_cols)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.5 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle("PDF i CDF svih audio feature-a po klasi", fontsize=14, fontweight="bold", y=1.005)

    for row, feat in enumerate(feature_cols):
        ax_pdf = axes[row][0]
        ax_cdf = axes[row][1]

        for i, klasa in enumerate(klase):
            vals = df_feat[df_feat["class"] == klasa][feat].dropna().values
            vals = vals[np.isfinite(vals)]
            if len(vals) < 5:
                continue
            color = class_color(i)

            # PDF (KDE ili fallback)
            x_min, x_max = vals.min(), vals.max()
            margin = (x_max - x_min) * 0.1 + 1e-9

            if np.std(vals) < 1e-10 or len(np.unique(vals)) < 3:
                ax_pdf.axvline(vals[0], color=color, linewidth=2,
                               linestyle="--", label=f"{klasa} (konst. {vals[0]:.4g})")
            else:
                try:
                    kde    = gaussian_kde(vals, bw_method="scott")
                    x_grid = np.linspace(x_min - margin, x_max + margin, 300)
                    y_kde  = kde(x_grid)
                    ax_pdf.plot(x_grid, y_kde, color=color, linewidth=1.8, label=klasa)
                    ax_pdf.fill_between(x_grid, y_kde, alpha=0.10, color=color)
                except Exception:
                    ax_pdf.hist(vals, bins=30, density=True, color=color,
                                alpha=0.45, label=f"{klasa} (hist)", histtype="stepfilled")
                    ax_pdf.hist(vals, bins=30, density=True, color=color,
                                alpha=0.9, histtype="step", linewidth=1.5)

            # CDF
            sv = np.sort(vals)
            cy = np.arange(1, len(sv) + 1) / len(sv)
            ax_cdf.plot(sv, cy, color=color, linewidth=1.8, label=klasa)

        ax_pdf.set_title(f"PDF — {feat}", fontsize=10, fontweight="bold")
        ax_pdf.set_xlabel(feat, fontsize=9)
        ax_pdf.set_ylabel("Gustoća", fontsize=9)
        ax_pdf.legend(fontsize=7, framealpha=0.85)
        ax_pdf.grid(True, alpha=0.2, linestyle="--")
        ax_pdf.spines["top"].set_visible(False)
        ax_pdf.spines["right"].set_visible(False)

        ax_cdf.set_title(f"CDF — {feat}", fontsize=10, fontweight="bold")
        ax_cdf.set_xlabel(feat, fontsize=9)
        ax_cdf.set_ylabel("P(X ≤ x)", fontsize=9)
        ax_cdf.set_ylim(0, 1.05)
        ax_cdf.legend(fontsize=7, framealpha=0.85)
        ax_cdf.grid(True, alpha=0.2, linestyle="--")
        ax_cdf.spines["top"].set_visible(False)
        ax_cdf.spines["right"].set_visible(False)
        for q, label in [(0.25, "Q1"), (0.50, "Q2"), (0.75, "Q3")]:
            ax_cdf.axhline(q, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {output_path}")


# 1. skeniranje dataseta
print("1. Skeniranje dataseta...")

meta = defaultdict(list)

for klasa in sorted(os.listdir(DATASET_DIR)):
    klasa_dir = Path(DATASET_DIR) / klasa
    if not klasa_dir.is_dir():
        continue

    for wav_file in klasa_dir.rglob("*.wav"):
        try:
            y, sr = librosa.load(str(wav_file), sr=None, mono=False)
            if y.ndim > 1:
                channels = y.shape[0]
                y = librosa.to_mono(y)
            else:
                channels = 1

            duration   = librosa.get_duration(y=y, sr=sr)
            rms        = float(np.sqrt(np.mean(y**2)))
            is_silent  = rms < 1e-4
            is_corrupt = False

        except Exception:
            duration, sr, channels, rms, is_silent = 0, 0, 0, 0, False
            is_corrupt = True

        meta[klasa].append({
            "file":      str(wav_file),
            "sr":        sr,
            "duration":  duration,
            "channels":  channels,
            "rms":       rms,
            "silent":    is_silent,
            "corrupt":   is_corrupt,
        })

    n         = len(meta[klasa])
    n_corrupt = sum(1 for m in meta[klasa] if m["corrupt"])
    n_silent  = sum(1 for m in meta[klasa] if m["silent"])
    print(f"  {klasa:10s}: {n:5d} fajlova | {n_corrupt} corrupt | {n_silent} silent")

klase = list(meta.keys())
boje  = [class_color(i) for i in range(len(klase))]



# 2. Distribucija klasa
print("\n2. Vizualizacija distribucije klasa...")

brojevi = [len(meta[k]) for k in klase]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bars = axes[0].bar(klase, brojevi, color=boje, edgecolor="white", linewidth=1.2)
axes[0].set_title("Broj uzoraka po klasi", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Broj fajlova")
axes[0].set_xlabel("Klasa vozila")
for bar, val in zip(bars, brojevi):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(val), ha="center", va="bottom", fontsize=9)

axes[1].pie(brojevi, labels=klase, colors=boje, autopct="%1.1f%%", startangle=140, pctdistance=0.8)
axes[1].set_title("Udio klasa (class balance)", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_distribucija_klasa.png", dpi=150, bbox_inches="tight")
plt.close()
print(f" Spremljeno: {OUTPUT_DIR}/01_distribucija_klasa.png")



# 3. Distribucija trajanja i sampling rate  +  PDF/CDF
print("\n3. Distribucija trajanja i sampling rate...")

all_durations = {k: [m["duration"] for m in meta[k] if not m["corrupt"]] for k in klase}
all_sr        = {k: [m["sr"]       for m in meta[k] if not m["corrupt"]] for k in klase}
all_rms       = {k: [m["rms"]      for m in meta[k] if not m["corrupt"] and not m["silent"]] for k in klase}

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for i, k in enumerate(klase):
    axes[0, 0].hist(all_durations[k], bins=30, alpha=0.55, label=k, color=boje[i])

axes[0, 0].set_title("Distribucija trajanja po klasi")
axes[0, 0].set_xlabel("Trajanje (s)")
axes[0, 0].set_ylabel("Broj uzoraka")
axes[0, 0].legend(fontsize=8)

data_dur = [all_durations[k] for k in klase]
bp = axes[0, 1].boxplot(data_dur, labels=klase, patch_artist=True)
for patch, c in zip(bp["boxes"], boje):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
axes[0, 1].set_title("Trajanje — boxplot po klasi")
axes[0, 1].set_ylabel("Trajanje (s)")

sr_summary = {}
for k in klase:
    srs = all_sr[k]
    unique, counts = np.unique(srs, return_counts=True)
    sr_summary[k] = dict(zip(unique.astype(int), counts))
    print(f"  {k:10s} SR vrijednosti: { {int(u): int(c) for u, c in zip(unique, counts)} }")

all_unique_sr = sorted(set(sr for k in klase for sr in sr_summary[k]))
x     = np.arange(len(klase))
width = 0.8 / max(len(all_unique_sr), 1)
for i, sr_val in enumerate(all_unique_sr):
    vals = [sr_summary[k].get(sr_val, 0) for k in klase]
    axes[1, 0].bar(x + i * width, vals, width, label=f"{sr_val} Hz", alpha=0.8)
axes[1, 0].set_xticks(x + width * len(all_unique_sr) / 2)
axes[1, 0].set_xticklabels(klase)
axes[1, 0].set_title("Sampling rate po klasi")
axes[1, 0].set_ylabel("Broj fajlova")
axes[1, 0].legend(fontsize=8)

mono_counts   = [sum(1 for m in meta[k] if m["channels"] == 1) for k in klase]
stereo_counts = [sum(1 for m in meta[k] if m["channels"] == 2) for k in klase]
axes[1, 1].bar(klase, mono_counts,   label="Mono",   color="#5DCAA5", alpha=0.8)
axes[1, 1].bar(klase, stereo_counts, bottom=mono_counts, label="Stereo", color="#7F77DD", alpha=0.8)
axes[1, 1].set_title("Mono vs Stereo po klasi")
axes[1, 1].set_ylabel("Broj fajlova")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_trajanje_sr.png", dpi=150, bbox_inches="tight")
plt.close()
print(f" Saved: {OUTPUT_DIR}/02_trajanje_sr.png")


print("\n3b. PDF/CDF — trajanje i RMS...")

plot_pdf_cdf(
    all_durations,
    feature_name="Trajanje uzorka",
    xlabel="Trajanje (s)",
    filename=f"{OUTPUT_DIR}/pdf_cdf/pdfcdf_trajanje.png"
)

plot_pdf_cdf(
    all_rms,
    feature_name="RMS glasnoća",
    xlabel="RMS amplituda",
    filename=f"{OUTPUT_DIR}/pdf_cdf/pdfcdf_rms.png",
    log_scale=True
)

plot_pdf_cdf(
    all_sr,
    feature_name="Sampling Rate",
    xlabel="Sample Rate (Hz)",
    filename=f"{OUTPUT_DIR}/pdf_cdf/pdfcdf_sr.png"
)


# 4. Provjera kvalitete
print("\n4. Provjera kvalitete...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

silent_counts  = [sum(1 for m in meta[k] if m["silent"])  for k in klase]
corrupt_counts = [sum(1 for m in meta[k] if m["corrupt"]) for k in klase]

axes[0].bar(klase, silent_counts,  color="#E24B4A", alpha=0.8)
axes[0].set_title("Tihi fajlovi po klasi (RMS < 1e-4)")
axes[0].set_ylabel("Broj fajlova")

axes[1].bar(klase, corrupt_counts, color="#BA7517", alpha=0.8)
axes[1].set_title("Korupirani fajlovi po klasi")
axes[1].set_ylabel("Broj fajlova")

rms_data = [[m["rms"] for m in meta[k] if not m["corrupt"] and not m["silent"]] for k in klase]
bp2 = axes[2].boxplot(rms_data, labels=klase, patch_artist=True)
for patch, c in zip(bp2["boxes"], boje):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
axes[2].set_title("RMS glasnoća po klasi")
axes[2].set_ylabel("RMS amplituda")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_kvaliteta.png", dpi=150, bbox_inches="tight")
plt.close()
print(f" Saved: {OUTPUT_DIR}/03_kvaliteta.png")


# 5. Waveform i mel-spektrogram po klasi
print("\n5. Waveform i mel-spektrogram po klasi...")

fig = plt.figure(figsize=(18, len(klase) * 3.5))
gs  = gridspec.GridSpec(len(klase), 3, figure=fig, hspace=0.6, wspace=0.35)

for row, klasa in enumerate(klase):
    uzorci = [m for m in meta[klasa] if not m["corrupt"] and not m["silent"]]
    if not uzorci:
        print(f"  {klasa}: nema valjanih uzoraka, preskačem")
        continue

    sample = uzorci[0]
    try:
        y, sr = librosa.load(sample["file"], sr=SR_TARGET, mono=True)
    except Exception as e:
        print(f"  {klasa}: greška pri čitanju — {e}")
        continue

    ax_w = fig.add_subplot(gs[row, 0])
    librosa.display.waveshow(y, sr=sr, ax=ax_w, color=boje[row], alpha=0.8)
    ax_w.set_title(f"{klasa} — waveform", fontsize=10, fontweight="bold")
    ax_w.set_xlabel("Vrijeme (s)")
    ax_w.set_ylabel("Amplituda")

    ax_m = fig.add_subplot(gs[row, 1])
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time",
                                   y_axis="mel", ax=ax_m, cmap="magma")
    ax_m.set_title(f"{klasa} — mel-spektrogram", fontsize=10, fontweight="bold")
    fig.colorbar(img, ax=ax_m, format="%+2.0f dB", pad=0.02)

    ax_c = fig.add_subplot(gs[row, 2])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    img2 = librosa.display.specshow(mfcc, sr=sr, x_axis="time",
                                    ax=ax_c, cmap="coolwarm")
    ax_c.set_title(f"{klasa} — MFCC ({N_MFCC} koef.)", fontsize=10, fontweight="bold")
    fig.colorbar(img2, ax=ax_c, pad=0.02)

    print(f"  {klasa:10s}: waveform + mel-spektrogram + MFCC — ok")

plt.suptitle("Vizualizacija zvuka po klasi vozila", fontsize=14, fontweight="bold", y=1.01)
plt.savefig(f"{OUTPUT_DIR}/04_vizualizacija_zvuka.png", dpi=150, bbox_inches="tight")
plt.close()
print(f" Saved: {OUTPUT_DIR}/04_vizualizacija_zvuka.png")


# 6. Feature selekcija + korelacija + PCA  +  PDF/CDF po svakom featureu
print("\n6. Feature ekstrakcija + PDF/CDF...")

feature_rows = []

for klasa in klase:
    uzorci = [m for m in meta[klasa] if not m["corrupt"] and not m["silent"]]

    for sample in uzorci[:100]:
        try:
            y, sr = librosa.load(sample["file"], sr=SR_TARGET, mono=True)

            mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr       = librosa.feature.zero_crossing_rate(y)
            rms       = librosa.feature.rms(y=y)
            flatness  = librosa.feature.spectral_flatness(y=y)
            contrast  = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            row = {
                "class":            klasa,
                "mfcc_mean":        float(np.mean(mfcc)),
                "mfcc_std":         float(np.std(mfcc)),
                "centroid_mean":    float(np.mean(centroid)),
                "bandwidth_mean":   float(np.mean(bandwidth)),
                "rolloff_mean":     float(np.mean(rolloff)),
                "zcr_mean":         float(np.mean(zcr)),
                "rms_mean":         float(np.mean(rms)),
                "flatness_mean":    float(np.mean(flatness)),
                "contrast_mean":    float(np.mean(contrast)),
                "onset_strength":   float(np.mean(onset_env)),
                "onset_std":        float(np.std(onset_env)),
                "duration":         sample["duration"],
            }
            feature_rows.append(row)

        except Exception:
            continue

df_feat = pd.DataFrame(feature_rows)
print(f"  Ekstrahirano {len(df_feat)} uzoraka za analizu")

# nazivi feature stupaca
feature_cols = [c for c in df_feat.columns if c != "class"]

# pdf/cdf svaki feature zasebno
print("\n  Generiranje PDF/CDF za svaki feature (zasebni fajlovi)...")
for feat in feature_cols:
    data_per_class = {
        k: df_feat[df_feat["class"] == k][feat].dropna().tolist()
        for k in klase
    }
    log = feat in ("rms_mean", "flatness_mean")
    plot_pdf_cdf(
        data_per_class,
        feature_name=feat,
        xlabel=feat,
        filename=f"{OUTPUT_DIR}/pdf_cdf/pdfcdf_{feat}.png",
        log_scale=log
    )

# pdf/cdf svi audio featurei zajedno
print("\n  Generiranje PDF/CDF grid (svi featurei zajedno)...")
plot_pdf_cdf_grid(
    df_feat,
    feature_cols=feature_cols,
    klase=klase,
    output_path=f"{OUTPUT_DIR}/pdf_cdf/pdfcdf_grid_svi_featurei.png"
)

# korelacijska mat
X = df_feat.drop(columns=["class"])
corr = X.corr()

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=ax)
plt.title("Korelacijska matrica feature-a", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_feature_korelacija.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> Saved: {OUTPUT_DIR}/05_feature_korelacija.png")

# pca
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca        = PCA()
X_pca      = pca.fit_transform(X_scaled)
explained  = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(explained) + 1), cum_explained, marker="o", color="#3498db")
axes[0].axhline(0.95, color="red", linestyle="--", alpha=0.7, label="95% prag")
axes[0].set_xlabel("Broj PCA komponenti")
axes[0].set_ylabel("Kumulativna objašnjena varijanca")
axes[0].set_title("PCA — kumulativna varijanca", fontsize=11, fontweight="bold")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(range(1, len(explained) + 1), explained, color="#9b59b6", alpha=0.8)
axes[1].set_xlabel("PCA komponenta")
axes[1].set_ylabel("Objašnjena varijanca")
axes[1].set_title("PCA — varijanca po komponenti", fontsize=11, fontweight="bold")
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> Saved: {OUTPUT_DIR}/06_pca.png")

# pca 2d scatter po klasi
fig, ax = plt.subplots(figsize=(9, 7))
for i, klasa in enumerate(klase):
    mask = df_feat["class"] == klasa
    ax.scatter(X_pca[mask.values, 0], X_pca[mask.values, 1],
               c=class_color(i), s=20, alpha=0.6, label=klasa)
ax.set_title("PCA 2D — feature prostor po klasi", fontsize=12, fontweight="bold")
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
ax.legend(fontsize=9, markerscale=2)
ax.grid(True, alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06b_pca_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> Saved: {OUTPUT_DIR}/06b_pca_scatter.png")


# DONE
print("\nEda analiza gotova. Svi grafovi spremljeni u:", OUTPUT_DIR)