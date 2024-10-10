from collections import defaultdict
import scipy.stats as sps
from utils.reload import load_data
from utils.fit_fig import fit_fig
from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from mave_calibration.main import prep_data
from utils import threshold_utils
from utils.skewnorm import get_cdf_dist
from fire import Fire
from tqdm import tqdm
import yaml
pd.set_option('display.max_columns', 500)

def load_dataset(dataset_id,**kwargs):
    processed_data_path = Path(kwargs.get("processed_data_path","/data/dzeiberg/mave_calibration/processed_datasets"))
    scores, sample_indicators, labels = load_data(processed_data_path / f"{dataset_id}.json", load_replicates=False)
    return scores, sample_indicators, labels

def load_results(dataset_id, result_sources,**kwargs):
    """results_dir = Path(results_dir)
    results = []
    sample_orders = set()
    for result_file in results_dir.glob(f"*/*{dataset_id}.json"):
        with open(result_file) as f:
            result = json.load(f)
            result['weights'] = np.array(result['weights'])
            sample_orders.add(tuple(result['sample_order']))
            results.append(result)
    if len(sample_orders) > 1:
        true_sample_order = sorted(list(sample_orders), key=lambda x: len(x))[-1]
        results = [result for result in results if tuple(result['sample_order']) == true_sample_order]
    return results"""
    INCLUDES_SYNONYMOUS = kwargs.get("includes_synonymous",True)
    DEBUG_NUM = kwargs.get("debug_num",None)
    results = []
    invalid_results = []
    for results_dir in result_sources:
        if DEBUG_NUM and len(results) > DEBUG_NUM:
            break
        results_dir = Path(results_dir)
        for result_file in results_dir.glob(f"*/*{dataset_id}.json"):
            with open(result_file) as f:
                result = json.load(f)
                if (INCLUDES_SYNONYMOUS and result['sample_order'] == ['P/LP','B/LB','gnomAD','synonymous']) or \
                    (not INCLUDES_SYNONYMOUS and result['sample_order'] == ['P/LP','B/LB','gnomAD']):
                    result['weights'] = np.array(result['weights'])
                    result['sample_indicators'] = np.array(result['sample_indicators'])
                    result['observations'] = np.array(result['observations'])
                    if not result['weights'].shape[0]:
                        raise ValueError(f"empty weights: {result_file}")
                        # print(f"empty weights: {result_file} {result['weights']}")
                        # continue
                    results.append(result)
                else:
                    invalid_results.append(result)
                    # print(f"invalid sample order: {result['sample_order']}")
            if DEBUG_NUM and len(results) > DEBUG_NUM:
                break
    if len(invalid_results):
        print(f"invalid results: {len(invalid_results)}")
    return results

def generate_fit_fig(scores,labels,sample_indicators,results,save_dir,Tau_p,Tau_b):
    # Fit fig
    fig,ax = plt.subplots(len(labels),1,figsize=(15,10),sharex=True,sharey=True)
    fit_fig(scores, sample_indicators, labels,
            ax.ravel()[:sample_indicators.shape[1]],
            results=results)
    for label, tau,linestyle in zip(["+1", "+2", "+3", "+4", "+8"],Tau_p,["dotted", "dashdot", "dashed", "solid","dotted"]):
            if np.isnan(tau):
                continue
            for axi in ax.ravel():
                axi.axvline(tau, color='red', linestyle=linestyle, alpha=.5)

    for label, tau,linestyle in zip(["-1", "-2", "-3", "-4", "-8"],Tau_b,["dotted", "dashdot", "dashed", "solid",'dotted']):
            if np.isnan(tau):
                continue
            for axi in ax.ravel():
                axi.axvline(tau, color='blue', linestyle=linestyle, alpha=.5)
    ax[-1].set_xlabel("Assay Score")
    fig.savefig(save_dir / "fit_fig.png",bbox_inches='tight',dpi=300)
    plt.close(fig)

def generate_dist_fig(scores,labels,sample_indicators,results,save_dir):
    # cdf dist fig
    distances = [np.array([get_cdf_dist(scores[sample_indicators[:,i]],
                                        result['component_params'],
                                        result['weights'][i]) for result in results])
                for i in range(len(labels))]
    fig,ax = plt.subplots(1,1)
    sns.boxplot(dict(zip([f"{l} (n={sample_indicators[:,i].sum():,d})" for i,l in enumerate(labels)],distances)),
                orient='h',ax=ax)
    ax.set_xlabel("Normalized Yang (P=2) distance")
    ax.set_xlim(0,max(np.concatenate(distances))*1.1)
    fig.savefig(save_dir / "cdf_dist.png",bbox_inches='tight',dpi=300)
    plt.close(fig)

def get_out_of_bag_observations(scoreset, result):
    bootstrap_indices = result['bootstrap_indices']
    candidates = scoreset[scoreset.labels.apply(lambda x: len(set(x).intersection(result['sample_order'])) > 0)]
    oob = np.ones(len(candidates),dtype=bool)
    oob[bootstrap_indices] = False
    return candidates[oob]

def get_oob_preds(scoreset, results, Tau_p, Tau_b,is_inverted):
    oob_preds = defaultdict(list)
    for result in results:
        oob = get_out_of_bag_observations(scoreset,result)
        for idx, r in oob.iterrows():
            oob_preds[idx].append(threshold_utils.assign_assay_evidence_strength(np.mean(r.scores),Tau_p, Tau_b, is_inverted=is_inverted))
    return {k : tuple(v) for k,v in oob_preds.items()}

def get_point_distr_by_class(ss,labels):
    point_distr = {}
    for label in labels:
        sc = ss[ss.labels.apply(lambda x: label in x)]
        if len(sc) == 0:
            continue
        point_distr[label] = sc.oob_pred_mode.value_counts().sort_index()
    df= pd.DataFrame(point_distr).fillna(0).T.astype(int)
    df.columns = df.columns.astype(int)
    return df

def generate_oob_fig(ss,results,save_dir,Tau_p,Tau_b,is_inverted,labels,scoreset_config):
    ss = ss.assign(oob_preds = get_oob_preds(ss,results,Tau_p,Tau_b,is_inverted))
    ss = ss.assign(oob_pred_mode=[sps.mode(x).mode.item() for x in ss.oob_preds])
    sns.heatmap(get_point_distr_by_class(ss,labels),cmap='Blues',annot=True,fmt='d',cbar_kws={'label': 'Count'})
    plt.ylabel("Class")
    plt.xlabel("Calibration Points")
    fig = plt.gcf()
    fig.savefig(save_dir / "point_distr_by_label.png",bbox_inches='tight',dpi=300)
    plt.close(fig)

    classification_column = scoreset_config.get("classification_column","author_labels")
    hm = ss.loc[:,[classification_column,'oob_pred_mode']].dropna().groupby(classification_column).oob_pred_mode.value_counts().unstack().fillna(0).astype(int)
    hm.index = hm.index.str.replace("_"," ")
    hm.columns = hm.columns.astype(int)
    sns.heatmap(hm,cmap='Blues',annot=True,fmt='d',cbar_kws={'label': 'Count'})
    plt.ylabel("Author Classification")
    plt.xlabel("Calibration Points")
    fig = plt.gcf()
    fig.savefig(save_dir / "point_distr_by_author_class.png",bbox_inches='tight',dpi=300)
    plt.close(fig)

    missing_oob = ss[(ss.oob_pred_mode.isna()) & (ss.labels.apply(lambda x: len(set(x).intersection(labels)) > 0))]
    missing_oob.to_csv(save_dir / "missing_oob.csv")

def generate_prior_fig(priors,save_dir):
    try:
        fig,ax = plt.subplots(1,1)
        ax.hist(priors,bins=100)
        ax.set_xlabel("Prior Probability Pathogenicity")
        fig.savefig(save_dir / "prior_hist.png",bbox_inches='tight',dpi=300)
        plt.close(fig)
    except ValueError:
        with open(save_dir / "priors.txt","w") as f:
            f.write("\n".join([str(x) for x in priors]))

def generate_dataset_figs(dataset_id, results_dirs,save_dir,scoreset_config,**kwargs):
    save_dir = Path(save_dir) / (dataset_id)
    save_dir.mkdir(exist_ok=True,parents=True)
    scores, sample_indicators, labels = load_dataset(dataset_id,**kwargs)
    data_filepath = Path(kwargs.get("processed_data_path","/data/dzeiberg/mave_calibration/processed_datasets")) / f"{dataset_id}.json"
    ss = pd.read_json(data_filepath)
    INCLUDES_SYNONYMOUS = "synonymous" in labels
    results = load_results(dataset_id,results_dirs,includes_synonymous=INCLUDES_SYNONYMOUS,**kwargs)
    if not len(results):
        print(f"No results found for {dataset_id}")
        return
    with open(save_dir / "num_results.txt","w") as f:
        f.write(f"{len(results)}")
    tauPFile = save_dir / "Tau_p.npy"
    tauBFile = save_dir / "Tau_b.npy"
    P_file = save_dir / "P.npy"
    B_file = save_dir / "B.npy"
    priors_file = save_dir / "priors.npy"
    is_inverted_file = save_dir / "is_inverted.npy"
    if tauPFile.exists() and \
        tauBFile.exists() and \
            P_file.exists() and \
                B_file.exists() and \
                    priors_file.exists() and \
                        is_inverted_file.exists():
        Tau_p = np.load(tauPFile)
        Tau_b = np.load(tauBFile)
        P = np.load(P_file)
        B = np.load(B_file)
        priors = np.load(priors_file)
        is_inverted = np.load(is_inverted_file)
    else:
        Tau_p, Tau_b, priors,is_inverted,P,B = threshold_utils.get_score_threshold(np.linspace(scores.min(),scores.max(),3000),results,)
        np.save(save_dir / "Tau_p.npy",Tau_p)
        np.save(save_dir / "Tau_b.npy",Tau_b)
        np.save(save_dir / "P.npy",P)
        np.save(save_dir / "B.npy",B)
        np.save(save_dir / "priors.npy",priors)
        np.save(save_dir / "is_inverted.npy",is_inverted)
    generate_prior_fig(priors,save_dir)
    generate_fit_fig(scores,labels,sample_indicators,results,save_dir,Tau_p,Tau_b)
    generate_dist_fig(scores,labels,sample_indicators,results,save_dir)
    generate_oob_fig(ss,results,save_dir,Tau_p,Tau_b, is_inverted, labels,scoreset_config)

def generate_all_figs(processed_data_path,results_dirs,save_dir,**kwargs):
    save_dir = Path(save_dir)
    processed_data_path = Path(processed_data_path)
    config_filepath = kwargs.get("config_filepath","/home/dzeiberg/mave_preprocessing/datasets.yaml")
    with open(config_filepath) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    data_files = list(processed_data_path.glob("*.json"))
    for dataset_id in tqdm(data_files,total=len(data_files)):
        dataset_id = dataset_id.stem
        dataset_name = dataset_id.split("_pipeline")[0]
        generate_dataset_figs(dataset_id,
                                results_dirs,
                                save_dir,
                                processed_data_path=processed_data_path,
                                scoreset_config=config[dataset_name],
                                **kwargs)
    
if __name__ == "__main__":
    # Fire(generate_all_figs)
    generate_all_figs("/Users/danielzeiberg/Desktop/processed_datasets",
                        [Path("/Users/danielzeiberg/Desktop/results_10_09_24"),],
                        "/Users/danielzeiberg/Desktop/figs_10_09_24",
                        config_filepath="/Users/danielzeiberg/Desktop/mave_results/datasets.yaml",)