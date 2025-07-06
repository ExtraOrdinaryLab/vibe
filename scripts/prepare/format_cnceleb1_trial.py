import os
import argparse

from tqdm import tqdm


def create_cnceleb_trails(cnceleb_root, trial_path, extension='flac'):
    """
    Create formatted trial file for CN-Celeb dataset evaluation.
    
    Args:
        cnceleb_root: Root directory of CN-Celeb dataset
        trial_path: Output path for the formatted trial file
        extension: Audio file extension (flac, wav, etc.)
    """
    # Path to enrollment list and trials list
    enroll_lst_path = os.path.join(cnceleb_root, "eval/lists/enroll.lst")
    raw_trl_path = os.path.join(cnceleb_root, "eval/lists/trials.lst")

    # Create speaker to audio file mapping
    spk2wav_mapping = {}
    
    # Load enrollment list with standard open()
    with open(enroll_lst_path, 'r') as f:
        enroll_lst = [line.strip().split() for line in f if line.strip()]
    
    for item in tqdm(enroll_lst, desc='Speaker mapping'):
        path = os.path.splitext(item[1])
        spk2wav_mapping[item[0]] = path[0] + '.{}'.format(extension)
    
    # Load trials data with standard open()
    with open(raw_trl_path, 'r') as f:
        trials = [line.strip().split() for line in f if line.strip()]

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(trial_path), exist_ok=True)

    # Write formatted trial file with label and audio paths
    with open(trial_path, "w") as f:
        for item in tqdm(trials, desc='Handle trials'):
            enroll_path = os.path.join(cnceleb_root, "eval", spk2wav_mapping[item[0]])
            test_path = os.path.join(cnceleb_root, "eval", item[1])
            test_path = os.path.splitext(test_path)[0] + '.{}'.format(extension)
            label = item[2]
            f.write("{} {} {}\n".format(label, enroll_path, test_path))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Format CN-Celeb1 trial list for speaker verification")
    parser.add_argument('--cnceleb_root', type=str, default='/home/jovyan/corpus/audio/cnceleb1/CN-Celeb_flac',
                        help='Root directory of CN-Celeb dataset')
    parser.add_argument('--trial_path', type=str, default='trials/cnceleb1_trial.txt',
                        help='Output path for the formatted trial list')
    parser.add_argument('--extension', type=str, default='flac',
                        help='Audio file extension (e.g., flac, wav)')
    
    args = parser.parse_args()
    
    # Call the processing function with parsed arguments
    create_cnceleb_trails(
        cnceleb_root=args.cnceleb_root, 
        trial_path=args.trial_path, 
        extension=args.extension
    )


if __name__ == "__main__":
    main()