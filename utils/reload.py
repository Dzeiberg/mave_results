import pandas as pd
import fire
import numpy as np
import json

def load_data(data_filepath,load_replicates=False,labels=["P/LP", "B/LB", 'gnomAD', 'synonymous']):
    """
    Load data from a json file.

    Each instance is assigned all its labels (not one-hot)
    if load_replicates is True, then each replicate is a separate sample,
        otherwise the average of replicates is used for the score

    Args:
    -----
        data_filepath: str, path to the json file
        load_replicates: bool, if True, load all replicates as separate samples, otherwise use average of replicates

    Returns:
    --------
        scores: np.array, the scores for each sample
        sample_indicators: np.array, the indicators for each sample
        labels: np.array, the labels for each indicator
    """
    data = pd.read_json(data_filepath)
    # labels = ["P/LP", "B/LB", 'gnomAD', 'synonymous']
    sample_indicators = []
    scores = []
    for i,(_,observation) in enumerate(data.iterrows()):
        indicator = [label in observation['labels'] for label in labels]
        if load_replicates:
            for score in observation['scores']:
                scores.append(score)
                sample_indicators.append(indicator)
        else:
            scores.append(np.mean(observation['scores']))
            sample_indicators.append(indicator)
    scores = np.array(scores)
    sample_indicators = np.array(sample_indicators)
    sample_mask = sample_indicators.sum(axis=0) > 0
    labels = np.array(labels)[sample_mask]
    sample_indicators = sample_indicators[:,sample_mask]
    return scores, sample_indicators,labels

def load_result(result_file):
    """
    Load the results from a json file.
    Args:
        result_file: str, path to the json file

    Returns:
        results: dict, the results
    """
    with open(result_file, "r") as f:
        results = json.load(f)
    return results

if __name__ == "__main__":
    fire.Fire(load_data)
    # data_filepath = "/data/dzeiberg/mave_calibration/processed_datasets/BAP1_SGE.json"
    # scores, sample_indicators = load_data(data_filepath)