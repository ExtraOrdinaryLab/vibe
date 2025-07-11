"""
Speaker Verification Evaluation Script.
This script computes cosine similarity scores between enrollment and test embeddings,
and calculates evaluation metrics such as EER and minDCF.
"""

import os
import sys
import re
import hashlib
import argparse
import numpy as np
from tqdm import tqdm
from kaldiio import ReadHelper
from sklearn.metrics.pairwise import cosine_similarity

from vibe.utils import get_logger
from vibe.evaluation import compute_pmiss_pfa_rbst, compute_eer, compute_c_norm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute speaker verification scores and metrics')
    parser.add_argument('--enrol_data', default='', type=str, 
                        help='Directory containing enrollment embeddings')
    parser.add_argument('--test_data', default='', type=str, 
                        help='Directory containing test embeddings')
    parser.add_argument('--scores_dir', default='', type=str, 
                        help='Directory to save scores and metrics')
    parser.add_argument('--trials', nargs='+', 
                        help='List of trial files to evaluate')
    parser.add_argument('--p_target', default=0.01, type=float, 
                        help='Prior probability of target trials in DCF calculation')
    parser.add_argument('--c_miss', default=1, type=float, 
                        help='Cost of miss detection in DCF calculation')
    parser.add_argument('--c_fa', default=1, type=float, 
                        help='Cost of false alarm in DCF calculation')
    
    return parser.parse_args(sys.argv[1:])


def sha256_hash(data):
    """
    Generate SHA-256 hash of input data.
    
    Args:
        data: String or bytes to hash
        
    Returns:
        Hexadecimal digest of the hash
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    sha256_hash_object = hashlib.sha256(data)
    return sha256_hash_object.hexdigest()


def collect_embeddings(data_dir):
    """
    Collect embeddings from Kaldi-format ark files.
    
    Args:
        data_dir: Directory containing embedding ark files
        
    Returns:
        Dictionary mapping keys to embedding arrays
    """
    data_dict = {}
    emb_arks = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if re.search('.ark$', i)]
    
    if len(emb_arks) == 0:
        raise Exception(f'No embedding ark files found in {data_dir}')

    # Load embeddings from all ark files
    for ark in emb_arks:
        with ReadHelper(f'ark:{ark}') as reader:
            for key, array in reader:
                data_dict[key] = array

    return data_dict


def process_trial(trial_path, enrol_dict, test_dict, scores_dir, p_target, c_miss, c_fa, logger):
    """
    Process a single trial file, compute scores and metrics.
    
    Args:
        trial_path: Path to the trial file
        enrol_dict: Dictionary of enrollment embeddings
        test_dict: Dictionary of test embeddings
        scores_dir: Directory to save scores
        p_target, c_miss, c_fa: Parameters for DCF calculation
        logger: Logger object for recording results
    """
    scores = []
    labels = []

    # Get trial name and prepare score file path
    trial_name = os.path.basename(trial_path)
    score_path = os.path.join(scores_dir, f'{trial_name}.score')
    
    # Process each line in the trial file
    with open(trial_path, 'r') as trial_f, open(score_path, 'w') as score_f:
        lines = trial_f.readlines()
        for line in tqdm(lines, desc=f'Scoring trial {trial_name}'):
            pair = line.strip().split()
            
            # Get embeddings using hashed keys
            enrol_emb = enrol_dict[sha256_hash(pair[1])]
            test_emb = test_dict[sha256_hash(pair[2])]
            
            # Compute cosine similarity score
            cosine_score = cosine_similarity(
                enrol_emb.reshape(1, -1),
                test_emb.reshape(1, -1)
            )[0][0]
            
            # Write score to output file
            score_f.write(' '.join(pair) + ' %.5f\n' % cosine_score)
            
            # Collect score and label for metrics computation
            scores.append(cosine_score)
            if pair[0] == '1' or pair[0] == 'target':
                labels.append(1)
            elif pair[0] == '0' or pair[0] == 'nontarget':
                labels.append(0)
            else:
                raise Exception(f'Unrecognized label in {line}.')

    # Convert to numpy arrays for efficient computation
    scores = np.array(scores)
    labels = np.array(labels)

    # Compute evaluation metrics
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, threshold = compute_eer(fnr, fpr, scores)
    min_dcf = compute_c_norm(
        fnr, 
        fpr, 
        p_target=p_target,
        c_miss=c_miss,
        c_fa=c_fa
    )

    # Log the metrics results
    logger.info(f"Results of {trial_name}:")
    logger.info(f"EER = {100 * eer:.4f}%")
    logger.info(f"minDCF (p_target:{p_target} c_miss:{c_miss} c_fa:{c_fa}) = {min_dcf:.4f}")


def main():
    """Main function to run the speaker verification evaluation."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.scores_dir, exist_ok=True)

    # Set up logging
    result_path = os.path.join(args.scores_dir, 'result.metrics')
    logger = get_logger(fpath=result_path, fmt="%(message)s")

    # Collect embeddings from enrollment and test directories
    enrol_dict = collect_embeddings(args.enrol_data)
    test_dict = collect_embeddings(args.test_data)

    # Process each trial file
    for trial in args.trials:
        process_trial(
            trial_path=trial,
            enrol_dict=enrol_dict,
            test_dict=test_dict,
            scores_dir=args.scores_dir,
            p_target=args.p_target,
            c_miss=args.c_miss,
            c_fa=args.c_fa,
            logger=logger
        )


if __name__ == "__main__":
    main()