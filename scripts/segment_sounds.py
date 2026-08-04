"""Rezanje wav datoteke u isjcke od po 2 sec

Outlieri (datoteke od kojih se ne uzima nista):
  - --max-duration: datoteke dulje od ovoga (u sekundama) se ne obraduju
    (preduge/vjerojatno pogresan sadrzaj)
  - --min-rms: apsolutni prag glasnoce (0-1); ako ni jedan prozor u datoteci
    ne prijede taj prag (cijela snimka je prakticki tisina/sum), datoteka daje
    0 isjecaka
  Takve se datoteke prijave na kraju, a fizicki se brisu samo uz
  --delete-outliers (zadano iskljuceno, jer je brisanje nepovratno).

Pokretanje (iz roota repoa):
  python scripts/segment_sounds.py --category scream
  python scripts/segment_sounds.py --category car_crash --max-segments 5
  python scripts/segment_sounds.py --category scream --energy-percentile 50 --dry-run
  python scripts/segment_sounds.py --category scream --max-duration 40 --min-rms 0.01 --delete-outliers ---> (ovo je najbolji izbor za outlier detekciju, ali brise datoteke, pa pazi)
"""

import argparse
import glob
import os
import sys

import librosa
import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCIDENTS_DIR = os.path.join(BASE, 'data', 'incidents')


def pick_segment_starts(y, sr, seg_len_sec, hop_frac, energy_percentile,
                         min_segments, max_segments, min_rms=None):
    """Vrati listu pocetnih indeksa (u uzorcima) ne-preklapajucih isjecaka,
    poredanih kronoloski, birajuci najenergicnije dijelove zvuka.

    Ako je min_rms zadan i nijedan prozor ga ne dosize, vraca praznu listu
    (datoteka je outlier - nema stvarnog sadrzaja iznad tog praga)."""
    seg_len = int(seg_len_sec * sr)
    if len(y) <= seg_len:
        if min_rms is not None:
            rms = np.sqrt(np.mean(y.astype(np.float64) ** 2))
            if rms < min_rms:
                return []
        return [0]

    hop = max(1, int(seg_len * hop_frac))
    starts = list(range(0, len(y) - seg_len + 1, hop))
    energies = np.array([
        np.sum(y[s:s + seg_len].astype(np.float64) ** 2) for s in starts
    ])
    rms_per_window = np.sqrt(energies / seg_len)

    valid = np.ones(len(starts), dtype=bool)
    if min_rms is not None:
        valid = rms_per_window >= min_rms
        if not valid.any():
            return []

    valid_idx = np.where(valid)[0]
    threshold = np.percentile(energies[valid_idx], energy_percentile)

    order = valid_idx[np.argsort(energies[valid_idx])[::-1]]
    chosen = []
    for idx in order:
        if len(chosen) >= max_segments:
            break
        if len(chosen) >= min_segments and energies[idx] < threshold:
            break
        s = starts[idx]
        if any(abs(s - c) < seg_len for c in chosen):
            continue
        chosen.append(s)

    chosen.sort()
    return chosen


def extract_segment(y, sr, start, seg_len_sec):
    seg_len = int(seg_len_sec * sr)
    segment = y[start:start + seg_len]
    if len(segment) < seg_len:
        segment = np.pad(segment, (0, seg_len - len(segment)))
    return segment


def existing_segment_count(out_dir, stem):
    return len(glob.glob(os.path.join(out_dir, f"{stem}_2s_*.wav")))


def process_file(path, out_dir, seg_len_sec, hop_frac, energy_percentile,
                  min_segments, max_segments, min_rms, max_duration,
                  overwrite, dry_run):
    """Vrati (n_segments, is_outlier). n_segments je 0 kad je datoteka outlier
    (predugacka ili bez sadrzaja iznad min_rms)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if not overwrite and existing_segment_count(out_dir, stem) > 0:
        return -1, False  # -1 = vec obradeno, ne outlier

    if max_duration is not None:
        duration = sf.info(path).duration
        if duration > max_duration:
            return 0, True

    y, sr = librosa.load(path, sr=None)
    starts = pick_segment_starts(
        y, sr, seg_len_sec, hop_frac, energy_percentile,
        min_segments, max_segments, min_rms=min_rms,
    )

    if not starts:
        return 0, True

    if dry_run:
        return len(starts), False

    for i, start in enumerate(starts):
        segment = extract_segment(y, sr, start, seg_len_sec)
        out_path = os.path.join(out_dir, f"{stem}_2s_{i:02d}.wav")
        sf.write(out_path, segment, sr)

    return len(starts), False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', required=True)
    parser.add_argument('--in-dir', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--segment-length', type=float, default=2.0)
    parser.add_argument('--hop-fraction', type=float, default=0.25)
    parser.add_argument('--energy-percentile', type=float, default=65.0)
    parser.add_argument('--min-segments', type=int, default=1)
    parser.add_argument('--max-segments', type=int, default=10)
    parser.add_argument('--min-rms', type=float, default=None)
    parser.add_argument('--max-duration', type=float, default=None)
    parser.add_argument('--delete-outliers', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.min_segments < 1:
        print("Greska: --min-segments mora biti >= 1.", file=sys.stderr)
        sys.exit(1)

    in_dir = args.in_dir or os.path.join(INCIDENTS_DIR, args.category)
    out_dir = args.out_dir or os.path.join(in_dir, 'segments_2s')

    if not os.path.isdir(in_dir):
        print(f"Greska: mapa ne postoji: {in_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    wav_files = sorted(glob.glob(os.path.join(in_dir, '*.wav')))
    print(f"Kategorija: {args.category}")
    print(f"Ulaz:  {in_dir} ({len(wav_files)} .wav datoteka)")
    print(f"Izlaz: {out_dir}")

    total_in = len(wav_files)
    total_segments = 0
    already_done = 0
    outliers = []

    for path in wav_files:
        n_segments, is_outlier = process_file(
            path, out_dir, args.segment_length, args.hop_fraction,
            args.energy_percentile, args.min_segments, args.max_segments,
            args.min_rms, args.max_duration, args.overwrite, args.dry_run,
        )
        if n_segments == -1:
            already_done += 1
            continue
        if is_outlier:
            outliers.append(path)
            continue
        total_segments += n_segments

    if outliers:
        action = "Brisem" if (args.delete_outliers and not args.dry_run) else "Outlieri (0 isjecaka)"
        print(f"\n{action} - {len(outliers)} datoteka bez sadrzaja iznad praga:")
        for path in outliers:
            print(f"  - {os.path.relpath(path, BASE)}")
        if args.delete_outliers and not args.dry_run:
            for path in outliers:
                os.remove(path)
        elif args.delete_outliers and args.dry_run:
            print("  (dry-run, nista nije stvarno obrisano)")

    print(f"\nGotovo. Ulaznih datoteka: {total_in}, vec obradeno ranije: "
          f"{already_done}, outlieri: {len(outliers)}, izlaznih isjecaka: "
          f"{total_segments}"
          f"{' (dry-run, nista nije spremljeno)' if args.dry_run else ''}.")


if __name__ == '__main__':
    main()
