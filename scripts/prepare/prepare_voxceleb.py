import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict
from functools import partial
from multiprocessing import Pool  # Replace concurrent.futures with multiprocessing

import librosa
from tqdm import tqdm
from rich.console import Console

console = Console()


def fast_scandir(path: str, extensions: List[str], recursive: bool = False):
    # Scan files recursively faster than glob
    # From github.com/drscotthawley/aeiou/blob/main/aeiou/core.py
    subfolders, files = [], []

    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(path):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in extensions:
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    if recursive:
        for path in list(subfolders):
            sf, f = fast_scandir(path, extensions, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f)  # type: ignore

    return subfolders, files


def extract_spk_id(file_path):
    file_path = Path(file_path)
    # Example: /voxceleb1/test/wav/id10273/Ed918auNI0Y/00005.wav
    speaker_id = file_path.parent.parent.name  # id10273
    youtube_id = file_path.parent.name         # Ed918auNI0Y
    utterance_id = file_path.name              # 00005.wav
    return speaker_id


def get_duration(file_path: str) -> float:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        raise ValueError(f"Can't open {file_path}")


def process_file(file_path):
    duration = get_duration(file_path)
    return {
        'audio': {'path': file_path},
        'spk_id': extract_spk_id(file_path),
        'duration': float(duration)
    }


def write_to_jsonl(file_paths, save_path, max_workers: int = 8):
    # Use multiprocessing.Pool instead of ProcessPoolExecutor for better performance with large batches
    with Pool(processes=max_workers) as pool:
        with open(save_path, 'w') as f:
            # Calculate optimal chunk size to reduce inter-process communication overhead
            chunksize = max(1, len(file_paths) // (max_workers * 4))
            # Use imap instead of map for better memory efficiency with streaming results
            for entry in tqdm(pool.imap(process_file, file_paths, chunksize=chunksize), 
                             total=len(file_paths), desc=f'Write to {save_path}'):
                f.write(json.dumps(entry) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Prepare VoxCeleb manifest")
    parser.add_argument("--voxceleb_root", type=str, default=None, required=True, help="")
    parser.add_argument("--manifest_root", type=str, default=None, required=True, help="")
    parser.add_argument("--manifest_prefix", type=str, default=None, required=True, help="")
    parser.add_argument("--max_workers", type=int, default=8, help="")
    args = parser.parse_args()

    voxceleb_root = args.voxceleb_root
    manifest_root = args.manifest_root
    manifest_prefix = args.manifest_prefix
    max_workers = args.max_workers

    os.makedirs(manifest_root, exist_ok=True)

    _, dev_wav_files = fast_scandir(
        os.path.join(voxceleb_root, 'dev', 'wav'), extensions=['.wav'], recursive=True
    )
    _, test_wav_files = fast_scandir(
        os.path.join(voxceleb_root, 'test', 'wav'), extensions=['.wav'], recursive=True
    )

    console.log(f"{len(dev_wav_files)} files in dev set")
    console.log(f"{len(test_wav_files)} files in test set")

    write_to_jsonl(
        dev_wav_files, 
        save_path=os.path.join(manifest_root, f'{manifest_prefix}_dev.jsonl'), 
        max_workers=max_workers
    )
    write_to_jsonl(
        test_wav_files, 
        save_path=os.path.join(manifest_root, f'{manifest_prefix}_test.jsonl'), 
        max_workers=max_workers
    )


if __name__ == "__main__":
    main()