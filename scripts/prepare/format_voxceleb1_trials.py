import os
import argparse
import urllib.request


def is_url(path):
    """Check if the given path is a URL."""
    return path.startswith(("http://", "https://"))


def read_trials(src_path):
    """Read trials from a local file or URL."""
    if is_url(src_path):
        # If it's a URL, download the content
        with urllib.request.urlopen(src_path) as response:
            content = response.read().decode('utf-8')
            lines = content.strip().split('\n')
    else:
        # If it's a local file
        with open(src_path, 'r') as file:
            lines = file.read().strip().split('\n')
    
    # Process the lines into trials
    trials = []
    for line in lines:
        parts = line.strip().split()
        # Each 3 elements form a group (label, enrollment audio, test audio)
        for i in range(0, len(parts), 3):
            if i+2 < len(parts):  # Ensure we have a complete triplet
                trials.append(parts[i:i+3])
    
    return trials


def build_speaker_location_map(voxceleb1_root):
    """Build a mapping from speaker ID to location (dev/test)"""
    speaker_location = {}
    
    # Check dev directory
    dev_path = os.path.join(voxceleb1_root, "dev", "wav")
    if os.path.exists(dev_path):
        for speaker_id in os.listdir(dev_path):
            if os.path.isdir(os.path.join(dev_path, speaker_id)):
                speaker_location[speaker_id] = "dev"
    
    # Check test directory
    test_path = os.path.join(voxceleb1_root, "test", "wav")
    if os.path.exists(test_path):
        for speaker_id in os.listdir(test_path):
            if os.path.isdir(os.path.join(test_path, speaker_id)):
                speaker_location[speaker_id] = "test"
    
    return speaker_location


def get_audio_path(voxceleb1_root, speaker_location, audio_path):
    """Get the full path to an audio file"""
    # Extract speaker ID from path
    parts = audio_path.split('/')
    speaker_id = parts[0]
    
    # Determine if the speaker is in dev or test directory
    location = speaker_location.get(speaker_id)
    if location:
        return os.path.join(voxceleb1_root, location, "wav", audio_path)
    else:
        # If location not found, try default paths
        dev_path = os.path.join(voxceleb1_root, "dev", "wav", audio_path)
        if os.path.exists(dev_path):
            return dev_path
        
        test_path = os.path.join(voxceleb1_root, "test", "wav", audio_path)
        if os.path.exists(test_path):
            return test_path
        
        # If still not found, return a possible path and log warning
        print(f"Warning: Could not find location for audio {audio_path}, using default path")
        return os.path.join(voxceleb1_root, "wav", audio_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb1_root', help='voxceleb1_root', type=str,
                        default="datasets/VoxCeleb/voxceleb1/")
    parser.add_argument('--src_trial_path', help='src_trials_path (local file or URL)', 
                        type=str, default="https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt")
    parser.add_argument('--dst_trial_path', help='dst_trials_path',
                        type=str, default="data/trial.lst")
    args = parser.parse_args()

    # Read trial data
    trials = read_trials(args.src_trial_path)
    
    # Build mapping from speaker ID to location
    speaker_location = build_speaker_location_map(args.voxceleb1_root)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.dst_trial_path), exist_ok=True)
    
    # Write formatted trials to output file
    with open(args.dst_trial_path, "w") as f:
        for item in trials:
            enroll_path = get_audio_path(args.voxceleb1_root, speaker_location, item[1])
            test_path = get_audio_path(args.voxceleb1_root, speaker_location, item[2])
            f.write(f"{item[0]} {enroll_path} {test_path}\n")


if __name__ == "__main__":
    main()
    """
    Available VoxCeleb trial list URLs:
    - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt (verification test set)
    - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt (hard test set)
    - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt (complete test set)
    """