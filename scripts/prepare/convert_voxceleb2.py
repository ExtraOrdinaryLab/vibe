import os
import argparse
import subprocess
from pathlib import Path
from typing import List
from multiprocessing import Pool  # Replace concurrent.futures with multiprocessing

from tqdm import tqdm
from rich.console import Console

console = Console()


def fast_scandir(path: str, extensions: List[str], recursive: bool = False):
    # Scan files recursively faster than glob
    subfolders, files = [], []

    try:  # Try to avoid 'permission denied' errors
        for f in os.scandir(path):
            try:  # Try to avoid 'too many levels of symbolic links' errors
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
            files.extend(f)

    return subfolders, files


def convert_m4a_to_wav(file_path: str):
    # Convert m4a audio file to wav format with 16kHz sampling rate
    src_path = Path(file_path)
    dst_path = Path(str(src_path).replace('/aac/', '/wav/')).with_suffix('.wav')

    if dst_path.exists():
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', str(src_path),
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        str(dst_path)
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def convert_files_in_parallel(file_list: List[str], max_workers: int = 4):
    # Convert files in parallel using multiprocessing.Pool for better performance with large batches
    with Pool(processes=max_workers) as pool:
        # Calculate optimal chunk size to reduce inter-process communication overhead
        chunksize = max(1, len(file_list) // (max_workers * 4))
        
        # Use imap_unordered for better performance as order doesn't matter for this task
        for _ in tqdm(
            pool.imap_unordered(convert_m4a_to_wav, file_list, chunksize=chunksize),
            total=len(file_list),
            desc="Converting"
        ):
            pass  # We don't need the return values, just tracking progress


def main():
    parser = argparse.ArgumentParser(description="Prepare VoxCeleb manifest")
    parser.add_argument("--voxceleb2_root", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    voxceleb2_root = args.voxceleb2_root
    max_workers = args.max_workers

    _, dev_wav_files = fast_scandir(
        os.path.join(voxceleb2_root, 'dev', 'aac'), extensions=['.m4a'], recursive=True
    )
    _, test_wav_files = fast_scandir(
        os.path.join(voxceleb2_root, 'test', 'aac'), extensions=['.m4a'], recursive=True
    )

    os.makedirs(os.path.join(voxceleb2_root, 'dev', 'wav'), exist_ok=True)
    os.makedirs(os.path.join(voxceleb2_root, 'test', 'wav'), exist_ok=True)

    console.log(f"{len(dev_wav_files)} files in dev set")
    console.log(f"{len(test_wav_files)} files in test set")

    assert len(dev_wav_files) == 1092009, f"Expected 1092009 files in dev, got {len(dev_wav_files)}"
    assert len(test_wav_files) == 36237, f"Expected 36237 files in test, got {len(test_wav_files)}"

    console.log("Converting dev set...")
    convert_files_in_parallel(dev_wav_files, max_workers)

    console.log("Converting test set...")
    convert_files_in_parallel(test_wav_files, max_workers)


if __name__ == "__main__":
    main()