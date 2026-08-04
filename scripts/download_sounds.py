"""
WEB SCRAPER
Trazi i skida zvukove za odabranu kategoriju s Freesound API-ja i sprema ih kao
.wav u data/incidents/<kategorija>/.

Upute;
  export FREESOUND_API_KEY="tvoj_key"

Pokretanje (iz roota), kategorija se bira preko --category:
  python scripts/download_sounds.py --category scream
  python scripts/download_sounds.py --category car_crash --max 200 (stavi na 300 to je okej za ove manje klase)
  python scripts/download_sounds.py --category explosion --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCIDENTS_DIR = os.path.join(BASE, 'data', 'incidents')
API_URL = 'https://freesound.org/apiv2/search/text/'
PAGE_SIZE = 150

# license kljuc 
LICENSE_MAP = {
    'cc0': 'publicdomain/zero',
    'by': 'licenses/by/',
    'by-nc': 'licenses/by-nc/',
}

# kategorija -> pojmovi za pretragu. Kategorija koje nema ovdje pretrazuje se
# samo po vlastitom imenu (podvlake -> razmaci), vidi queries_for_category().
QUERY_PRESETS = {
    'scream': [
        'scream', 'screaming', 'woman scream', 'man scream',
        'screaming person', 'terrified scream', 'horror scream', 'yell shout',
    ],
    'explosion': [
        'explosion', 'explosion sound effect', 'bomb explosion', 'blast',
    ],
    'car_crash': [
        'car crash', 'car accident', 'vehicle collision', 'crash impact',
        'car crash sound effect',
    ],
    'gun_shot': [
        'gun shot', 'gunfire', 'pistol shot', 'rifle shot',
    ],
    'glass_breaking': [
        'glass breaking', 'glass shatter',
    ],
    'siren': [
        'siren', 'police siren', 'ambulance siren',
    ],
}


def queries_for_category(category):
    return QUERY_PRESETS.get(category, [category.replace('_', ' ')])


def load_credits(credits_path):
    """Vrati set vec skinutih Freesound ID-eva citajuci CREDITS.txt."""
    seen = set()
    if not os.path.exists(credits_path):
        return seen
    with open(credits_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ' by ' not in line:
                continue
            soundid = line.split(' by ', 1)[0].strip()
            if soundid.isdigit():
                seen.add(int(soundid))
    return seen


def append_credit(credits_path, soundid, username):
    is_new = not os.path.exists(credits_path)
    with open(credits_path, 'a', encoding='utf-8') as f:
        if is_new:
            f.write("Zvukovi skinuti s Freesound-a (scripts/download_sounds.py).\n\n")
            f.write("to access user page:  https://freesound.org/people/[username]\n")
            f.write("to access sound page: https://freesound.org/people/[username]/sounds/[soundid]\n\n")
            f.write("[soundid] by [username]\n\n")
        f.write(f"{soundid} by {username}\n")


def slugify(name):
    keep = [c if c.isalnum() else '-' for c in name.lower()]
    slug = ''.join(keep)
    while '--' in slug:
        slug = slug.replace('--', '-')
    return slug.strip('-')[:60]


def search_page(query, api_key, page, allowed_license_substrings):
    params = {
        'query': query,
        'token': api_key,
        'fields': 'id,name,username,license,previews,duration',
        'page_size': PAGE_SIZE,
        'page': page,
    }
    resp = requests.get(API_URL, params=params, timeout=30)
    if resp.status_code == 429:
        print("  Rate limit (429), cekam 10s...")
        time.sleep(10)
        return search_page(query, api_key, page, allowed_license_substrings)
    resp.raise_for_status()
    return resp.json()


def license_allowed(license_url, allowed_substrings):
    return any(s in license_url for s in allowed_substrings)


def download_and_convert(sound, out_dir, tmp_dir):
    preview_url = None
    previews = sound.get('previews', {})
    for key in ('preview-hq-mp3', 'preview-lq-mp3'):
        if previews.get(key):
            preview_url = previews[key]
            break
    if not preview_url:
        return False

    soundid = sound['id']
    slug = slugify(sound['name'])
    base_name = f"freesound_community-{slug}-{soundid}"
    wav_path = os.path.join(out_dir, base_name + '.wav')
    if os.path.exists(wav_path):
        return False

    tmp_mp3 = os.path.join(tmp_dir, base_name + '.mp3')
    try:
        r = requests.get(preview_url, timeout=60)
        r.raise_for_status()
    except requests.RequestException:
        print(f"Skip: {preview_url}")
        return False
    with open(tmp_mp3, 'wb') as f:
        f.write(r.content)

    result = subprocess.run(
        ['ffmpeg', '-y', '-i', tmp_mp3, '-ar', '44100', '-ac', '1', wav_path,
         '-loglevel', 'error'],
        capture_output=True,
    )
    os.remove(tmp_mp3)
    if result.returncode != 0:
        print(f"  ffmpeg greska za {base_name}: {result.stderr.decode(errors='replace')}")
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', required=True)
    parser.add_argument('--api-key', default=os.environ.get('FREESOUND_API_KEY'))
    parser.add_argument('--queries', default=None)
    parser.add_argument('--licenses', default='cc0')
    parser.add_argument('--max', type=int, default=500)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if not args.api_key:
        print("Greska: nedostaje Freesound API key. Postavi FREESOUND_API_KEY ili "
              "koristi --api-key.", file=sys.stderr)
        sys.exit(1)

    allowed_keys = [k.strip() for k in args.licenses.split(',') if k.strip()]
    unknown = [k for k in allowed_keys if k not in LICENSE_MAP]
    if unknown:
        print(f"Greska: nepoznata licenca(e) {unknown}, dostupno: {list(LICENSE_MAP)}",
              file=sys.stderr)
        sys.exit(1)
    allowed_substrings = [LICENSE_MAP[k] for k in allowed_keys]

    out_dir = args.out_dir or os.path.join(INCIDENTS_DIR, args.category)
    os.makedirs(out_dir, exist_ok=True)
    credits_path = os.path.join(out_dir, 'CREDITS.txt')
    seen_ids = load_credits(credits_path)
    print(f"Kategorija: {args.category}  (izlazna mapa: {out_dir})")
    print(f"Vec skinuto (iz CREDITS.txt): {len(seen_ids)} zvukova")

    if args.queries:
        queries = [q.strip() for q in args.queries.split(',') if q.strip()]
    else:
        queries = queries_for_category(args.category)
    downloaded = 0
    tmp_dir = os.path.join(out_dir, '_tmp_downloads')
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        for query in queries:
            if downloaded >= args.max:
                break
            print(f"\nPretrazujem: '{query}'")
            page = 1
            while downloaded < args.max:
                data = search_page(query, args.api_key, page, allowed_substrings)
                results = data.get('results', [])
                if not results:
                    break
                for sound in results:
                    if downloaded >= args.max:
                        break
                    soundid = sound['id']
                    if soundid in seen_ids:
                        continue
                    if not license_allowed(sound.get('license', ''), allowed_substrings):
                        continue
                    seen_ids.add(soundid)

                    if args.dry_run:
                        print(f"  [dry-run] {soundid} - {sound['name']} by {sound['username']}")
                        downloaded += 1
                        continue

                    ok = download_and_convert(sound, out_dir, tmp_dir)
                    if ok:
                        append_credit(credits_path, soundid, sound['username'])
                        downloaded += 1
                        if downloaded % 10 == 0:
                            print(f"  ... {downloaded} skinuto")
                        time.sleep(0.2)
                if not data.get('next'):
                    break
                page += 1
    finally:
        if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)

    print(f"\nGotovo. Novo skinuto: {downloaded} (max je bio {args.max}).")
    print(f"Credits azurirani u: {credits_path}")


if __name__ == '__main__':
    main()
