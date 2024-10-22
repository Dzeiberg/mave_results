import pandas as pd
import fire
import numpy as np
import json
from ast import literal_eval

def load_data(data_filepath,labels=["P/LP", "B/LB", 'gnomAD', 'synonymous', 'VUS'],):
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
    data = pd.read_csv(data_filepath).assign(labels=lambda x: x.labels.apply(literal_eval))
    sample_indicators = []
    scores = []
    hgvs_p = data.hgvs_p.values
    for i,(_,observation) in enumerate(data.iterrows()):
        indicator = [label in observation['labels'] for label in labels]
        scores.append(np.mean(observation['auth_reported_score']))
        sample_indicators.append(indicator)
    scores = np.array(scores)
    sample_indicators = np.array(sample_indicators)
    sample_mask = sample_indicators.sum(axis=0) > 0
    labels = np.array(labels)[sample_mask]
    sample_indicators = sample_indicators[:,sample_mask]
    author_labels = infer_author_labels(data)
    return scores, sample_indicators,labels, author_labels,hgvs_p

def infer_author_labels(data):
    """
    Infer the author labels from the data
    """
    if 'auth_reported_func_class' in data.columns and not data.auth_reported_func_class.isna().all():
        return data.auth_reported_func_class.values
    normal_min, normal_max, abnormal_min, abnormal_max = data.loc[:,['auth_reported_normal_min',
                                                                             'auth_reported_normal_max',
                                                                             'auth_reported_abnormal_min',
                                                                             'auth_reported_abnormal_max']].values[0]
    if not np.isnan([normal_min, abnormal_max]).any() and abnormal_max < normal_min:
        # standard score thresholds
        author_labels = []
        for score in data.auth_reported_score:
            if score >= normal_min:
                author_labels.append('Functionally Normal')
            elif score <= abnormal_max:
                author_labels.append('Functionally Abnormal')
            else:
                author_labels.append('Indeterminate')
    elif not np.isnan([normal_max, abnormal_min]).any() and abnormal_min > normal_max:
        # inverted score thresholds
        author_labels = []
        for score in data.auth_reported_score:
            if score <= normal_max:
                author_labels.append('Functionally Normal')
            elif score >= abnormal_min:
                author_labels.append('Functionally Abnormal')
            else:
                author_labels.append('Indeterminate')
    else:
        author_labels = ['N/a']*len(data)
    return np.array(author_labels)

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