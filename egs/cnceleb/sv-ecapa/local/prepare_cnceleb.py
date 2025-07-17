import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm


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
    # Example: /cnceleb1/CN-Celeb_wav/data/id00184/interview-02-037.wav
    speaker_id = file_path.parent.name  # id00184
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
            for entry in tqdm(
                pool.imap(
                    process_file, file_paths, chunksize=chunksize
                ), 
                total=len(file_paths), 
                desc=f'Write to {save_path}'
            ):
                f.write(json.dumps(entry) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Prepare CNCeleb manifest")
    parser.add_argument("--cnceleb1_root", type=str, default=None, required=True, help="")
    parser.add_argument("--cnceleb2_root", type=str, default=None, required=True, help="")
    parser.add_argument("--manifest_root", type=str, default=None, required=True, help="")
    parser.add_argument("--manifest_prefix", type=str, default="cnceleb", required=True, help="")
    parser.add_argument("--max_workers", type=int, default=8, help="")
    args = parser.parse_args()

    cnceleb1_root = args.cnceleb1_root
    cnceleb2_root = args.cnceleb2_root
    manifest_root = args.manifest_root
    manifest_prefix = args.manifest_prefix
    max_workers = args.max_workers

    os.makedirs(manifest_root, exist_ok=True)

    _, cnceleb1_files = fast_scandir(
        os.path.join(cnceleb1_root, 'CN-Celeb_wav', 'data'), extensions=['.wav'], recursive=True
    )
    _, cnceleb2_files = fast_scandir(
        os.path.join(cnceleb2_root, 'CN-Celeb2_wav', 'data'), extensions=['.wav'], recursive=True
    )

    cnceleb1_speakers = list(
        map(
            str.strip, 
            open(os.path.join(cnceleb1_root, 'CN-Celeb_flac', 'dev', 'dev.lst'), encoding='utf-8').read().splitlines()
        )
    )
    cnceleb2_speakers = list(
        map(
            str.strip, 
            open(os.path.join(cnceleb2_root, 'CN-Celeb2_flac', 'spk.lst'), encoding='utf-8').read().splitlines()
        )
    )
    allowed_speakers = cnceleb1_speakers + cnceleb2_speakers

    cnceleb1_files = [file for file in cnceleb1_files if extract_spk_id(file) in allowed_speakers]
    cnceleb2_files = [file for file in cnceleb2_files if extract_spk_id(file) in allowed_speakers]
    cnceleb_files = cnceleb1_files + cnceleb2_files
    print(f"{len(cnceleb1_files)} files in cnceleb1")
    print(f"{len(cnceleb2_files)} files in cnceleb2")

    write_to_jsonl(
        cnceleb_files, 
        save_path=os.path.join(manifest_root, f'{manifest_prefix}_dev.jsonl'), 
        max_workers=max_workers
    )


if __name__ == "__main__":
    main()