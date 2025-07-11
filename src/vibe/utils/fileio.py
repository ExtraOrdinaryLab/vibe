from typing import Dict


def load_wav_scp(file_path: str) -> Dict[str, str]:
    """
    Load a wav.scp file containing key-value pairs of audio IDs and file paths.
    
    Args:
        file_path: Path to the wav.scp file.
        
    Returns:
        Dictionary mapping audio IDs to file paths.
    """
    with open(file_path) as f:
        rows = [line.strip() for line in f.readlines()]
        result = {line.split()[0]: line.split()[1] for line in rows}
    return result