import os
import argparse


def read_trials(src_path):
    """Read trials from a local file."""
    # Read trials from local file
    with open(src_path, 'r') as file:
        lines = file.read().strip().split('\n')
    
    # Process the lines into trials
    trials = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:  # Each line contains label, enrollment audio, test audio
            trials.append(parts)
    
    return trials


def get_audio_path(vietnamceleb_root, audio_path):
    """Get the full path to an audio file"""
    # For VietnamCeleb, audio files are located in the 'data' directory
    full_path = os.path.join(vietnamceleb_root, "data", audio_path)
    
    # Check if the path exists
    if os.path.exists(full_path):
        return full_path
    else:
        # If not found, return the path and log warning
        print(f"Warning: Could not find audio file at {full_path}")
        return full_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vietnamceleb_root', help='vietnamceleb_root', type=str,
                        default="/path/to/VietnamCeleb/")
    parser.add_argument('--src_trial_path', help='src_trial_path (e.g., vietnam-celeb-e.txt)', 
                        type=str, default="vietnam-celeb-e.txt")
    parser.add_argument('--dst_trial_path', help='dst_trial_path',
                        type=str, default="data/trial.lst")
    args = parser.parse_args()

    # If src_trial_path is not absolute, assume it's in the vietnamceleb_root
    if not os.path.isabs(args.src_trial_path):
        src_full_path = os.path.join(args.vietnamceleb_root, args.src_trial_path)
    else:
        src_full_path = args.src_trial_path

    # Read trial data
    trials = read_trials(src_full_path)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.dst_trial_path), exist_ok=True)
    
    # Write formatted trials to output file
    with open(args.dst_trial_path, "w") as f:
        for item in trials:
            enroll_path = get_audio_path(args.vietnamceleb_root, item[1])
            test_path = get_audio_path(args.vietnamceleb_root, item[2])
            f.write(f"{item[0]} {enroll_path} {test_path}\n")


if __name__ == "__main__":
    main()
    """
    VietnamCeleb trial lists (located in the VietnamCeleb root directory):
    - vietnam-celeb-e.txt
    - vietnam-celeb-h.txt
    """