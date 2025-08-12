"""
Speaker Verification Evaluation Script.
This script computes cosine similarity scores between enrollment and test embeddings,
and calculates evaluation metrics such as EER and minDCF.
It also supports various score normalization techniques: z-norm, t-norm, s-norm, and as-norm.
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
    # Add score normalization arguments
    parser.add_argument('--score_norm', nargs='+', default=[], choices=['z-norm', 't-norm', 's-norm', 'as-norm'],
                        help='Score normalization techniques to apply (z-norm, t-norm, s-norm, as-norm)')
    parser.add_argument('--cohort_data', default='', type=str,
                        help='Directory containing cohort embeddings for score normalization')
    parser.add_argument('--top_n', default=300, type=int,
                        help='Number of top cohort embeddings to use for adaptive score normalization')
    
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


def apply_znorm(scores, cohort_scores):
    """
    Apply Z-norm to the scores.
    
    Args:
        scores: Original scores
        cohort_scores: Scores of the cohort speakers
        
    Returns:
        Z-normalized scores
    """
    # Compute mean and standard deviation of cohort scores
    mean = np.mean(cohort_scores)
    std = np.std(cohort_scores)
    
    # Apply Z-norm
    return (scores - mean) / std


def apply_tnorm(scores, cohort_scores):
    """
    Apply T-norm to the scores.
    
    Args:
        scores: Original scores
        cohort_scores: Scores of the cohort speakers
        
    Returns:
        T-normalized scores
    """
    # Compute mean and standard deviation of cohort scores for each test utterance
    mean = np.mean(cohort_scores, axis=0)
    std = np.std(cohort_scores, axis=0)
    
    # Apply T-norm
    return (scores - mean) / std


def apply_snorm(scores, z_cohort_scores, t_cohort_scores):
    """
    Apply S-norm to the scores.
    
    Args:
        scores: Original scores
        z_cohort_scores: Z-norm cohort scores
        t_cohort_scores: T-norm cohort scores
        
    Returns:
        S-normalized scores
    """
    # Apply Z-norm
    z_scores = apply_znorm(scores, z_cohort_scores)
    
    # Apply T-norm
    t_scores = apply_tnorm(scores, t_cohort_scores)
    
    # S-norm is the average of Z-norm and T-norm
    return 0.5 * (z_scores + t_scores)


def apply_adaptive_snorm(scores, z_cohort_scores, t_cohort_scores, top_n=300):
    """
    Apply Adaptive S-norm to the scores.
    
    Args:
        scores: Original scores
        z_cohort_scores: Z-norm cohort scores (enrol vs cohort) shape: [num_trials, num_cohort]
        t_cohort_scores: T-norm cohort scores (cohort vs test) shape: [num_cohort, num_trials]
        top_n: Number of top cohort scores to use
        
    Returns:
        Adaptive S-normalized scores
    """
    num_trials = len(scores)
    num_cohort = z_cohort_scores.shape[1]  # Number of cohort samples
    
    # Initialize arrays for adaptive statistics
    z_means = np.zeros(num_trials)
    z_stds = np.zeros(num_trials)
    t_means = np.zeros(num_trials)
    t_stds = np.zeros(num_trials)
    
    # Ensure top_n doesn't exceed the number of cohort samples
    top_n = min(top_n, num_cohort)
    
    # For each trial, find the top-n most similar cohort speakers
    for i in range(num_trials):
        # Get Z-norm (enrollment vs cohort) scores for this trial
        z_trial_scores = z_cohort_scores[i]
        # Sort and take top_n highest scores
        z_sorted_indices = np.argsort(z_trial_scores)[-top_n:]
        z_top = z_trial_scores[z_sorted_indices]
        
        # Compute Z-norm statistics using only top_n scores
        z_means[i] = np.mean(z_top)
        z_stds[i] = np.std(z_top)
        
        # Get T-norm (cohort vs test) scores for this trial
        t_trial_scores = t_cohort_scores[:, i]
        # Sort and take top_n highest scores
        t_sorted_indices = np.argsort(t_trial_scores)[-top_n:]
        t_top = t_trial_scores[t_sorted_indices]
        
        # Compute T-norm statistics using only top_n scores
        t_means[i] = np.mean(t_top)
        t_stds[i] = np.std(t_top)
    
    # Avoid division by zero
    z_stds = np.maximum(z_stds, 1e-8)
    t_stds = np.maximum(t_stds, 1e-8)
    
    # Apply AS-norm
    z_scores = (scores - z_means) / z_stds
    t_scores = (scores - t_means) / t_stds
    
    return 0.5 * (z_scores + t_scores)


def compute_cohort_scores(enrol_embs, test_embs, cohort_embs, logger):
    """
    Compute scores between enrollment/test embeddings and cohort embeddings.
    
    Args:
        enrol_embs: Dictionary of enrollment embeddings
        test_embs: Dictionary of test embeddings
        cohort_embs: Dictionary of cohort embeddings
        logger: Logger object
        
    Returns:
        z_cohort_dict: Dictionary mapping enrollment keys to cohort scores
        t_cohort_dict: Dictionary mapping test keys to cohort scores
    """
    # Get distributed processing information
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    # Convert dictionaries to arrays for faster computation
    enrol_keys = list(enrol_embs.keys())
    test_keys = list(test_embs.keys())
    cohort_keys = list(cohort_embs.keys())
    
    enrol_array = np.vstack([enrol_embs[key] for key in enrol_keys])
    test_array = np.vstack([test_embs[key] for key in test_keys])
    cohort_array = np.vstack([cohort_embs[key] for key in cohort_keys])
    
    # Compute Z-norm cohort scores (enrollment vs cohort)
    if rank == 0:
        logger.info(f"Computing Z-norm cohort scores for {len(enrol_keys)} enrollment embeddings against {len(cohort_keys)} cohort embeddings...")
    z_cohort_scores = cosine_similarity(enrol_array, cohort_array)
    
    # Compute T-norm cohort scores (cohort vs test)
    if rank == 0:
        logger.info(f"Computing T-norm cohort scores for {len(test_keys)} test embeddings against {len(cohort_keys)} cohort embeddings...")
    t_cohort_scores = cosine_similarity(cohort_array, test_array)
    
    # Create dictionaries mapping keys to cohort scores
    z_cohort_dict = {enrol_keys[i]: z_cohort_scores[i] for i in range(len(enrol_keys))}
    t_cohort_dict = {test_keys[i]: t_cohort_scores[:, i] for i in range(len(test_keys))}
    
    return z_cohort_dict, t_cohort_dict


def process_trial(trial_path, enrol_dict, test_dict, scores_dir, p_target, c_miss, c_fa, 
                  logger, score_norm=None, z_cohort_dict=None, t_cohort_dict=None, top_n=300):
    """
    Process a single trial file, compute scores and metrics.
    
    Args:
        trial_path: Path to the trial file
        enrol_dict: Dictionary of enrollment embeddings
        test_dict: Dictionary of test embeddings
        scores_dir: Directory to save scores
        p_target, c_miss, c_fa: Parameters for DCF calculation
        logger: Logger object for recording results
        score_norm: List of score normalization techniques to apply
        z_cohort_dict: Dictionary of Z-norm cohort scores
        t_cohort_dict: Dictionary of T-norm cohort scores
        top_n: Number of top cohort scores to use for adaptive normalization
    """
    # Get distributed processing information
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    scores = []
    labels = []
    
    # Lists to store enrollment and test keys in order
    enrol_keys = []
    test_keys = []

    # Get trial name and prepare score file path
    trial_name = os.path.basename(trial_path)
    score_file_prefix = os.path.join(scores_dir, f'{trial_name}')
    
    # Process each line in the trial file to collect keys and compute raw scores
    with open(trial_path, 'r') as trial_f:
        lines = trial_f.readlines()
        for line in tqdm(lines, desc=f'Processing trial {trial_name}', disable=rank!=0):
            pair = line.strip().split()
            
            # Get embeddings using hashed keys
            enrol_key = sha256_hash(pair[1])
            test_key = sha256_hash(pair[2])
            
            # Store keys for later use
            enrol_keys.append(enrol_key)
            test_keys.append(test_key)
            
            # Compute cosine similarity score
            enrol_emb = enrol_dict[enrol_key]
            test_emb = test_dict[test_key]
            cosine_score = cosine_similarity(
                enrol_emb.reshape(1, -1),
                test_emb.reshape(1, -1)
            )[0][0]
            
            # Collect score and label for metrics computation
            scores.append(cosine_score)
            if pair[0] == '1' or pair[0] == 'target':
                labels.append(1)
            elif pair[0] == '0' or pair[0] == 'nontarget':
                labels.append(0)
            else:
                raise Exception(f'Unrecognized label in {line}.')
    
    # Convert to numpy arrays for efficient computation
    raw_scores = np.array(scores)
    labels = np.array(labels)
    
    # Process raw scores (no normalization)
    save_scores_and_compute_metrics(
        scores=raw_scores,
        labels=labels,
        trial_lines=lines,
        output_path=f"{score_file_prefix}.score",
        trial_name=trial_name,
        norm_type="raw",
        p_target=p_target,
        c_miss=c_miss,
        c_fa=c_fa,
        logger=logger
    )
    
    # Apply score normalization if requested
    if score_norm and z_cohort_dict and t_cohort_dict:
        # Prepare cohort scores for each trial
        z_cohort_scores = np.array([z_cohort_dict[key] for key in enrol_keys])
        t_cohort_scores = np.array([t_cohort_dict[key] for key in test_keys]).T
        
        # Apply each requested normalization technique
        if 'z-norm' in score_norm:
            if rank == 0:
                logger.info(f"Applying Z-norm to {trial_name}...")
            z_norm_scores = apply_znorm(raw_scores, z_cohort_scores)
            save_scores_and_compute_metrics(
                scores=z_norm_scores,
                labels=labels,
                trial_lines=lines,
                output_path=f"{score_file_prefix}.znorm.score",
                trial_name=trial_name,
                norm_type="Z-norm",
                p_target=p_target,
                c_miss=c_miss,
                c_fa=c_fa,
                logger=logger
            )
            
        if 't-norm' in score_norm:
            if rank == 0:
                logger.info(f"Applying T-norm to {trial_name}...")
            t_norm_scores = apply_tnorm(raw_scores, t_cohort_scores)
            save_scores_and_compute_metrics(
                scores=t_norm_scores,
                labels=labels,
                trial_lines=lines,
                output_path=f"{score_file_prefix}.tnorm.score",
                trial_name=trial_name,
                norm_type="T-norm",
                p_target=p_target,
                c_miss=c_miss,
                c_fa=c_fa,
                logger=logger
            )
            
        if 's-norm' in score_norm:
            if rank == 0:
                logger.info(f"Applying S-norm to {trial_name}...")
            s_norm_scores = apply_snorm(raw_scores, z_cohort_scores, t_cohort_scores)
            save_scores_and_compute_metrics(
                scores=s_norm_scores,
                labels=labels,
                trial_lines=lines,
                output_path=f"{score_file_prefix}.snorm.score",
                trial_name=trial_name,
                norm_type="S-norm",
                p_target=p_target,
                c_miss=c_miss,
                c_fa=c_fa,
                logger=logger
            )
            
        if 'as-norm' in score_norm:
            if rank == 0:
                logger.info(f"Applying AS-norm (top {top_n}) to {trial_name}...")
            as_norm_scores = apply_adaptive_snorm(raw_scores, z_cohort_scores, t_cohort_scores, top_n)
            save_scores_and_compute_metrics(
                scores=as_norm_scores,
                labels=labels,
                trial_lines=lines,
                output_path=f"{score_file_prefix}.asnorm.score",
                trial_name=trial_name,
                norm_type=f"AS-norm (top {top_n})",
                p_target=p_target,
                c_miss=c_miss,
                c_fa=c_fa,
                logger=logger
            )


def save_scores_and_compute_metrics(scores, labels, trial_lines, output_path, 
                                   trial_name, norm_type, p_target, c_miss, c_fa, logger):
    """
    Save scores to file and compute evaluation metrics.
    
    Args:
        scores: Array of scores
        labels: Array of labels
        trial_lines: Original lines from the trial file
        output_path: Path to save the scores
        trial_name: Name of the trial
        norm_type: Description of the normalization applied
        p_target, c_miss, c_fa: Parameters for DCF calculation
        logger: Logger object for recording results
    """
    # Get distributed processing information
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    # Save scores to file
    with open(output_path, 'w') as score_f:
        for i, line in enumerate(trial_lines):
            pair = line.strip().split()
            score_f.write(' '.join(pair) + ' %.5f\n' % scores[i])
    
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

    # Log the metrics results (only on rank 0)
    if rank == 0:
        logger.info(f"Results of {trial_name} ({norm_type}):")
        logger.info(f"EER = {100 * eer:.4f}%")
        logger.info(f"minDCF (p_target:{p_target} c_miss:{c_miss} c_fa:{c_fa}) = {min_dcf:.4f}")


def main():
    """Main function to run the speaker verification evaluation."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Get distributed processing information
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.scores_dir, exist_ok=True)

    # Set up logging
    result_path = os.path.join(args.scores_dir, 'result.metrics')
    logger = get_logger(fpath=result_path, fmt="%(message)s")

    # Collect embeddings from enrollment and test directories
    if rank == 0:
        logger.info("Loading enrollment embeddings...")
    enrol_dict = collect_embeddings(args.enrol_data)
    
    if rank == 0:
        logger.info("Loading test embeddings...")
    test_dict = collect_embeddings(args.test_data)
    
    # Collect cohort embeddings if score normalization is requested
    z_cohort_dict = None
    t_cohort_dict = None
    if args.score_norm and args.cohort_data:
        if rank == 0:
            logger.info("Loading cohort embeddings for score normalization...")
        cohort_dict = collect_embeddings(args.cohort_data)
        
        if rank == 0:
            logger.info("Computing cohort scores for normalization...")
        z_cohort_dict, t_cohort_dict = compute_cohort_scores(
            enrol_dict, test_dict, cohort_dict, logger
        )
    elif args.score_norm and not args.cohort_data:
        if rank == 0:
            logger.warning("Score normalization requested but no cohort data provided. Skipping normalization.")
        args.score_norm = []

    # Process each trial file
    for trial in args.trials:
        if rank == 0:
            logger.info(f"Processing trial file: {trial}")
        process_trial(
            trial_path=trial,
            enrol_dict=enrol_dict,
            test_dict=test_dict,
            scores_dir=args.scores_dir,
            p_target=args.p_target,
            c_miss=args.c_miss,
            c_fa=args.c_fa,
            logger=logger,
            score_norm=args.score_norm,
            z_cohort_dict=z_cohort_dict,
            t_cohort_dict=t_cohort_dict,
            top_n=args.top_n
        )


if __name__ == "__main__":
    main()