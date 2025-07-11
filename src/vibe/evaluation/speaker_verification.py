"""
This script computes the official performance metrics for the NIST 2016 SRE.
The metrics include EER and DCFs (min/act).
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def compute_norm_counts(scores, edges, wghts=None):
    """
    Computes normalized (and optionally weighted) score counts for the bin edges.
    
    Args:
        scores: Array of scores
        edges: Bin edges for histogram
        wghts: Optional weights for the scores
        
    Returns:
        Normalized cumulative counts
    """
    if scores.size > 0:
        score_counts = np.histogram(scores, bins=edges, weights=wghts)[0].astype('f')
        norm_counts = np.cumsum(score_counts) / score_counts.sum()
    else:
        norm_counts = None
    return norm_counts


def compute_pmiss_pfa(scores, labels, weights=None):
    """
    Computes false positive rate (FPR) and false negative rate (FNR)
    given trial scores and their labels.
    
    Args:
        scores: Array of trial scores
        labels: Binary labels (1 for target, 0 for impostor)
        weights: Optional weights to equalize counts over score partitions
        
    Returns:
        fnr: False negative rate
        fpr: False positive rate
    """
    # Extract target and impostor scores
    tgt_scores = scores[labels == 1]  # Target trial scores
    imp_scores = scores[labels == 0]  # Impostor trial scores

    # Determine resolution for score binning
    resol = max([
        np.count_nonzero(labels == 0),  # Number of impostor trials
        np.count_nonzero(labels == 1),  # Number of target trials
        1.e6                            # Minimum resolution
    ])
    edges = np.linspace(np.min(scores), np.max(scores), resol)

    # Extract weights if provided
    if weights is not None:
        tgt_weights = weights[labels == 1]
        imp_weights = weights[labels == 0]
    else:
        tgt_weights = None
        imp_weights = None

    # Compute false negative rate and false positive rate
    fnr = compute_norm_counts(tgt_scores, edges, tgt_weights)
    fpr = 1 - compute_norm_counts(imp_scores, edges, imp_weights)

    return fnr, fpr


def compute_pmiss_pfa_rbst(scores, labels, weights=None):
    """
    Robust version for computing false positive rate (FPR) and false negative rate (FNR)
    given trial scores and their labels.
    
    Args:
        scores: Array of trial scores
        labels: Binary labels (1 for target, 0 for impostor)
        weights: Optional weights for trials
        
    Returns:
        fnr: False negative rate
        fpr: False positive rate
    """
    # Sort scores and reorder labels and weights accordingly
    sorted_ndx = np.argsort(scores)
    labels = labels[sorted_ndx]
    
    if weights is not None:
        weights = weights[sorted_ndx]
    else:
        weights = np.ones((labels.shape), dtype='f8')

    # Calculate target and impostor weights
    tgt_wghts = weights * (labels == 1).astype('f8')
    imp_wghts = weights * (labels == 0).astype('f8')

    # Compute false negative rate and false positive rate
    fnr = np.cumsum(tgt_wghts) / np.sum(tgt_wghts)
    fpr = 1 - np.cumsum(imp_wghts) / np.sum(imp_wghts)
    
    return fnr, fpr


def compute_eer(fnr, fpr, scores=None):
    """
    Computes the equal error rate (EER) given FNR and FPR values.
    
    Args:
        fnr: Array of false negative rates
        fpr: Array of false positive rates
        scores: Optional array of scores to return the threshold
        
    Returns:
        EER value and optionally the threshold at which it occurs
    """
    # Find the points where FNR and FPR cross
    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    
    # Interpolate to find the exact EER point
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))
    eer = fnr[x1] + a * (fnr[x2] - fnr[x1])

    if scores is not None:
        # Also return the score threshold at EER point
        score_sort = np.sort(scores)
        return eer, score_sort[x1]

    return eer


def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """
    Computes normalized minimum detection cost function (DCF).
    
    Args:
        fnr: Array of false negative rates
        fpr: Array of false positive rates
        p_target: A priori probability of target speakers
        c_miss: Cost of a miss (false negative)
        c_fa: Cost of a false alarm (false positive)
        
    Returns:
        Normalized minimum detection cost
    """
    # Calculate the detection cost
    c_det = min(c_miss * fnr * p_target + c_fa * fpr * (1 - p_target))
    
    # Calculate the default cost
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    
    # Return the normalized cost
    return c_det / c_def


def compute_c_dcf(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """
    Computes minimum detection cost function (DCF).
    
    Args:
        fnr: Array of false negative rates
        fpr: Array of false positive rates
        p_target: A priori probability of target speakers
        c_miss: Cost of a miss (false negative)
        c_fa: Cost of a false alarm (false positive)
        
    Returns:
        Minimum detection cost
    """
    # Calculate and return the minimum detection cost
    c_det = min(c_miss * fnr * p_target + c_fa * fpr * (1 - p_target))
    return c_det


def plot_det_curve(fnr, fpr, save_path=None):
    """
    Plots the detection error trade-off (DET) curve.
    
    Args:
        fnr: Array of false negative rates
        fpr: Array of false positive rates
        save_path: Optional path to save the plot image
    """
    # Transform FNR and FPR to probit scale for DET curve
    p_miss = norm.ppf(fnr)
    p_fa = norm.ppf(fpr)

    # Define tick marks and labels for the plot
    xytick = [
        0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
        0.2, 0.4
    ]
    xytick_labels = [str(x * 100) for x in xytick]

    # Create the DET plot
    plt.plot(p_fa, p_miss, 'r')
    plt.xticks(norm.ppf(xytick), xytick_labels)
    plt.yticks(norm.ppf(xytick), xytick_labels)
    plt.xlim(norm.ppf([0.00051, 0.5]))
    plt.ylim(norm.ppf([0.00051, 0.5]))
    plt.xlabel("False-alarm rate [%]", fontsize=12)
    plt.ylabel("False-reject rate [%]", fontsize=12)
    
    # Compute EER and mark it on the plot
    eer = compute_eer(fnr, fpr)
    plt.plot(norm.ppf(eer), norm.ppf(eer), 'o')
    
    # Add annotation for EER point
    plt.annotate(
        f"EER = {eer * 100:.2f}%",
        xy=(norm.ppf(eer), norm.ppf(eer)),
        xycoords='data',
        xytext=(norm.ppf(eer + 0.05), norm.ppf(eer + 0.05)),
        textcoords='data',
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3, rad=+0.2",
            fc="w"
        ),
        size=12,
        va='center',
        ha='center',
        bbox=dict(boxstyle="round4", fc="w"),
    )
    
    # Add grid for better readability
    plt.grid()
    
    # Save or display the plot
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def compute_equalized_scores(max_tar_imp_counts, sc, labs, masks):
    """
    Computes equalized scores for different trial partitions.
    
    Args:
        max_tar_imp_counts: Maximum counts for targets and impostors
        sc: Array of scores
        labs: Array of labels (1 for target, 0 for impostor)
        masks: List of boolean masks for different partitions
        
    Returns:
        Tuple of (scores, labels, count_weights)
    """
    count_weights = []
    scores = []
    labels = []
    
    # Process each partition defined by masks
    for ix in range(len(masks)):
        amask = masks[ix]
        alabs = labs[amask]
        
        # Count targets and non-targets in this partition
        num_targets = np.count_nonzero(alabs == 1)
        num_non_targets = alabs.size - num_targets
        
        # Store labels and scores for this partition
        labels.append(alabs)
        scores.append(sc[amask])
        
        # Calculate weights to equalize the contribution of each partition
        tar_weight = max_tar_imp_counts[0] / num_targets if num_targets > 0 else 0
        imp_weight = max_tar_imp_counts[1] / num_non_targets if num_non_targets > 0 else 0

        # Assign weights to each trial
        acount_weights = np.empty(alabs.shape, dtype='f')
        acount_weights[alabs == 1] = np.array([tar_weight] * num_targets)
        acount_weights[alabs == 0] = np.array([imp_weight] * num_non_targets)
        count_weights.append(acount_weights)

    # Combine results from all partitions
    scores = np.hstack(scores)
    labels = np.hstack(labels)
    count_weights = np.hstack(count_weights)

    return scores, labels, count_weights