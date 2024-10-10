import sys
sys.path.append("..")
from mave_results.utils.skewnorm import joint_densities
from mave_results.utils.evidence_framework import get_tavtigian_constant
from typing import List, Tuple
import logging
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def get_score_threshold(observations, results, control_sample_idx=1,parallel=True,return_all=False):
    """
    Get the score threshold for a given control sample index.
    
    Args:
        observations (np.array): The observations to use for the threshold calculation.
        result (dict): The result dictionary from the fit method.
        control_sample_idx (int): The index of the control sample in the result dictionary.
        
    Returns:
        float: The score threshold.
    """
    Tau_P = []
    Tau_B = []
    is_inverted = False
    _isflipped = [result["weights"][0,0] < result["weights"][1,0] for result in results]
    if np.sum(_isflipped) > len(results) / 2:
        is_inverted = True
    priors = np.array([prior_from_weights(np.array(result["weights"]), controls_idx=control_sample_idx,inverted=is_inverted) for result in results])
    valid_priors = priors[np.isfinite(priors)]
    if len(valid_priors) == 0:
        logging.error("No valid priors found")
        return np.ones(5) * np.nan, np.ones(5) * np.nan, priors, is_inverted
    minprior = np.nanmin(valid_priors)
    maxprior = np.nanmax(valid_priors)
    priors[priors <= 0] = minprior
    priors[priors >= 1] = maxprior

    def do_res(result,prior):
        log_lrPlus = get_log_lrPlus(observations, control_sample_idx, result)
        
        pathogenic_score_thresholds, benign_score_thresholds = calculate_score_thresholds(log_lrPlus,
                                                                                            prior, 
                                                                                            observations,
                                                                                            inverted=is_inverted)
        return pathogenic_score_thresholds, benign_score_thresholds
    if parallel:
        Tau_results = Parallel(n_jobs=-1,verbose=100)(delayed(do_res)(result,prior) for result,prior in zip(results,priors))
    else:
        Tau_results = [do_res(result,prior) for result,prior in tqdm(zip(results,priors))]
    Tau_P, Tau_B = zip(*Tau_results)
    P = np.stack(Tau_P)
    B = np.stack(Tau_B)
    if return_all:
        return P,B, priors,is_inverted, P,B
    # no more than ceiling(5%) of the bootstrap iterations can fail to meet the threshold
    maxFails = int(np.ceil(.05 * len(P)))
    meetsP = np.isnan(P).sum(0) < maxFails
    meetsB = np.isnan(B).sum(0) < maxFails
    # get the 5th percentile of the pathogenic scores and the 95th percentile of the benign scores (conservative thresholds)
    if is_inverted:
        Pscores = np.nanquantile(P,.95,axis=0)
        Bscores = np.nanquantile(B,.05,axis=0)
    else:
        Pscores = np.nanquantile(P,.05,axis=0)
        Bscores = np.nanquantile(B,.95,axis=0)
    Pscores[~meetsP] = np.nan
    
    Bscores[~meetsB] = np.nan
    return Pscores, Bscores, priors,is_inverted, P, B

def calculate_score_thresholds(log_LR,prior,rng,inverted=False):
    clipped_prior = np.clip(prior,.005,.55) # these seem to be the limits of the tavtigian constant
    lr_thresholds_pathogenic , lr_thresholds_benign = thresholds_from_prior(prior=clipped_prior,point_values=[1,2,4,8])
    log_lr_thresholds_pathogenic = np.log(lr_thresholds_pathogenic)
    log_lr_thresholds_benign = np.log(lr_thresholds_benign)
    pathogenic_score_thresholds = np.ones(len(log_lr_thresholds_pathogenic)) * np.nan
    benign_score_thresholds = np.ones(len(log_lr_thresholds_benign)) * np.nan
    for strength_idx,log_lr_threshold in enumerate(log_lr_thresholds_pathogenic):
        if log_lr_threshold is np.nan:
            continue
        exceed = np.where(log_LR > log_lr_threshold)[0]
        if len(exceed):
            if inverted:
                pathogenic_score_thresholds[strength_idx] = rng[min(exceed)]
            else:
                pathogenic_score_thresholds[strength_idx] = rng[max(exceed)]
    for strength_idx,log_lr_threshold in enumerate(log_lr_thresholds_benign):
        if log_lr_threshold is np.nan:
            continue
        exceed = np.where(log_LR < log_lr_threshold)[0]
        if len(exceed):
            if inverted:
                benign_score_thresholds[strength_idx] = rng[max(exceed)]
            else:
                benign_score_thresholds[strength_idx] = rng[min(exceed)]
    return pathogenic_score_thresholds,benign_score_thresholds

def get_log_lrPlus(X, control_sample_index, result, pathogenic_sample_num=0):
    f_P = joint_densities(X, result["component_params"],
                                        result["weights"][pathogenic_sample_num]).sum(0)
    f_B = joint_densities(X, result["component_params"],
                                        result["weights"][control_sample_index]).sum(0)
    return np.log(f_P) - np.log(f_B)
    

def get_priors(results, control_sample_index,inverted=False):
    priors = []
    for result in results:
        priors.append(prior_from_weights(np.array(result["weights"]),
                                    controls_idx=control_sample_index,inverted=inverted))
    priors = np.array(priors)
    # fill in nans/infs with median
    priors[np.isnan(priors) | np.isinf(priors)] = np.nanquantile(priors,.5)
    return priors

def prior_from_weights(weights : np.ndarray, population_idx : int=2, controls_idx : int=1, pathogenic_idx : int=0, inverted: bool = False) -> float:
    """
    Calculate the prior probability of an observation from the population being pathogenic

    Required Arguments:
    --------------------------------
    weights -- Ndarray (NSamples, NComponents)
        The mixture weights of each sample

    Optional Arguments:
    --------------------------------
    population_idx -- int (default 2)
        The index of the population component in the weights matrix
    
    controls_idx -- int (default 1)
        The index of the controls (i.e. benign) component in the weights matrix

    pathogenic_idx -- int (default 0)
        The index of the pathogenic component in the weights matrix

    Returns:
    --------------------------------
    prior -- float
        The prior probability of an observation from the population being pathogenic
    """
    if inverted:
        w_idx = 1
    else:
        w_idx = 0
    prior = ((weights[population_idx, w_idx] - weights[controls_idx, w_idx]) / (weights[pathogenic_idx, w_idx] - weights[controls_idx, w_idx])).item()
    if prior < 0:
        return -np.inf
    if prior > 1:
        return np.inf
    return prior


def thresholds_from_prior(prior, point_values=[1,2,3,4,8]) -> Tuple[List[float]]:
    """
    Get the evidence thresholds (LR+ values) for each point value given a prior

    Parameters
    ----------
    prior : float
        The prior probability of pathogenicity

    
    """
    exp_vals = 1 / np.array(point_values).astype(float)
    C,num_successes = get_tavtigian_constant(prior,return_success_count=True)
    # max number of successes is 17
    max_successes = 17
    pathogenic_evidence_thresholds = np.ones(len(point_values)) * np.nan
    benign_evidence_thresholds = np.ones(len(point_values)) * np.nan
    if num_successes < max_successes:
        logging.warning(f"Only ({num_successes})/{max_successes} rules for combining evidence are satisfied by constant {C}, found using prior of ({prior:.4f})")
        return pathogenic_evidence_thresholds, benign_evidence_thresholds
        
    for strength_idx, exp_val in enumerate(exp_vals):
        pathogenic_evidence_thresholds[strength_idx] = C ** exp_val
        benign_evidence_thresholds[strength_idx] = C ** -exp_val
    return pathogenic_evidence_thresholds[::-1], benign_evidence_thresholds[::-1]

def assign_assay_evidence_strength(score, pathogenic_score_thresholds, benign_score_thresholds,is_inverted=False):
    if np.isnan(score):
        return 0
    for threshold,points in list(zip(pathogenic_score_thresholds,[1,2,3,4,8]))[::-1]:
        if np.isnan(threshold):
            continue
        if is_inverted and score >= threshold:
            return points
        if (not is_inverted) and score <= threshold:
            return points
    for threshold,points in list(zip(benign_score_thresholds,[-1,-2,-3,-4,-8]))[::-1]:
        if np.isnan(threshold):
            continue
        if is_inverted and score <= threshold:
            return points
        if (not is_inverted) and score >= threshold:
            return points
    return 0

if __name__ == "__main__":
    import json
    from pathlib import Path
    results = []
    results_dir = Path("/data/dzeiberg/mave_calibration/results_10_06_24")
    dataset_id = "Jia_MSH2_SSM"
    pipeline = "A"
    INCLUDES_SYNONYMOUS = False
    for result_file in results_dir.glob(f"*/*{dataset_id}_pipeline_{pipeline}.json"):
        
        with open(result_file) as f:
            result = json.load(f)
            if (INCLUDES_SYNONYMOUS and result['sample_order'] == ['P/LP','B/LB','gnomAD','synonymous']) or (not INCLUDES_SYNONYMOUS):
                result['weights'] = np.array(result['weights'])
                results.append(result)
    get_score_threshold(np.arange(-5,5,.01),results,1,parallel=False)