import librosa
import numpy as np
import os
from tqdm import tqdm

# ─── settings ────────────────────────────────────────────────
SR             = 22050
CHUNK_DURATION = 30
CHUNK_SIZE     = SR * CHUNK_DURATION
N_FFT          = 2048
HOP_LENGTH     = 512

RAW_DIR        = os.path.expanduser("~/auralens/data/raw")
PROCESSED_DIR  = os.path.expanduser("~/auralens/data/processed")
COMPOSERS      = ["bach", "vivaldi", "paganini", "tchaikovsky"]
# ─────────────────────────────────────────────────────────────


def process_file(filepath, out_dir, filename_base):
    # get total duration without loading the whole file
    duration = librosa.get_duration(path=filepath)
    
    saved = 0
    start = 0
    i = 0
    
    while start + CHUNK_DURATION <= duration:
        # load only 30 seconds at a time — never the whole file
        chunk = librosa.load(filepath, sr=SR, 
                             offset=start, 
                             duration=CHUNK_DURATION)[0]
        
        D         = librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_LENGTH)
        magnitude = np.abs(D)
        
        out_path = os.path.join(out_dir, f"{filename_base}_chunk_{i:03d}.npy")
        np.save(out_path, magnitude)
        
        saved += 1
        i     += 1
        start += CHUNK_DURATION
    
    return saved


def main():
    total_chunks  = 0
    failed_files  = []

    for composer in COMPOSERS:
        in_dir  = os.path.join(RAW_DIR, composer)
        out_dir = os.path.join(PROCESSED_DIR, composer)
        os.makedirs(out_dir, exist_ok=True)

        mp3_files = [f for f in os.listdir(in_dir) if f.endswith(".mp3")]
        print(f"\n── {composer.upper()} ({len(mp3_files)} files) ──")

        for filename in tqdm(mp3_files):
            filepath      = os.path.join(in_dir, filename)
            filename_base = os.path.splitext(filename)[0]
            filename_base = filename_base.replace(" ", "_")[:50]

            # skip if already processed
            existing = [f for f in os.listdir(out_dir)
                        if f.startswith(filename_base)]
            if existing:
                print(f"  skipping {filename_base} — already processed")
                continue

            # process with error handling
            try:
                chunks_saved  = process_file(filepath, out_dir, filename_base)
                total_chunks += chunks_saved
                print(f"  ✓ {filename_base} — {chunks_saved} chunks")
            except Exception as e:
                print(f"  ✗ FAILED: {filename_base} — {e}")
                failed_files.append(filepath)

    print(f"\n══════════════════════════════")
    print(f"total chunks saved: {total_chunks}")

    if failed_files:
        print(f"\nfailed files ({len(failed_files)}):")
        for f in failed_files:
            print(f"  {f}")
    else:
        print("no failed files.")


if __name__ == "__main__":
    main()