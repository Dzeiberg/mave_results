from collections import defaultdict
from itertools import cycle
import scipy.stats as sps
from utils.reload import load_data
from utils.fit_fig import fit_fig
from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
import seaborn as sns
from utils import threshold_utils
from utils.skewnorm import get_cdf_dist
from fire import Fire
from tqdm import tqdm,trange
import yaml
from joblib import Parallel,delayed
pd.set_option('display.max_columns', 500)

def load_dataset(dataset_id,**kwargs):
    processed_data_path = Path(kwargs.get("processed_data_path","/data/dzeiberg/IGVF-cvfg-pillar-project/Pillar_project_data_files/individual_datasets/"))
    scores, sample_indicators, labels,author_labels,hgvs_p = load_data(processed_data_path / f"{dataset_id}.csv")
    return scores, sample_indicators, labels,author_labels,hgvs_p

def load_results(dataset_id, result_sources,**kwargs):
    INCLUDES_SYNONYMOUS = kwargs.get("includes_synonymous",True)
    DEBUG_NUM = kwargs.get("debug_num",None)
    results = []
    result_paths = []
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
                    result_paths.append(result_file)
                else:
                    invalid_results.append(result)
                    # print(f"invalid sample order: {result['sample_order']}")
            if DEBUG_NUM and len(results) > DEBUG_NUM:
                break
    if len(invalid_results):
        print(f"invalid results: {len(invalid_results)}")
    return results, result_paths

def generate_fit_fig(scores,labels,sample_indicators,results,save_dir,Tau_p,Tau_b,thresholds_to_plot=np.array([1,2,4,8]),priors=[]):
    # Fit fig
    fig,ax = plt.subplots(len(labels),1,figsize=(15,10),sharex=True,sharey=True)
    fit_fig(scores, sample_indicators, labels,
            ax.ravel()[:sample_indicators.shape[1]],
            results=results,priors=priors)
    threshold_included = np.array([x in thresholds_to_plot for x in np.arange(1,9)])
    Tau_p_plot = Tau_p[threshold_included]
    Tau_b_plot = Tau_b[threshold_included]
    line_styles = ["-","--","-.",":"]
    linecycler = cycle(line_styles)
    for label, tau in zip([f"+{int(p)}" for p in thresholds_to_plot],Tau_p_plot):
        if np.isnan(tau):
            continue
        ls = next(linecycler)
        for axi in ax.ravel():
            axi.axvline(tau, color='red', linestyle=ls, alpha=.5)

    linecycler = cycle(line_styles)
    for label, tau in zip([f"-{int(p)}" for p in thresholds_to_plot],Tau_b_plot):
        if np.isnan(tau):
            continue
        ls = next(linecycler)
        for axi in ax.ravel():
            axi.axvline(tau, color='blue', linestyle=ls, alpha=.5)
    ax[-1].set_xlabel("Assay Score")
    fig.savefig(save_dir / "fit_fig.png",bbox_inches='tight',dpi=300)
    plt.close(fig)

def generate_dist_fig(scores,labels,sample_indicators,results,save_dir):
    # cdf dist fig
    # distances = [np.array([get_cdf_dist(scores[sample_indicators[:,i]],
    #                                     result['component_params'],
    #                                     result['weights'][i]) for result in tqdm(results,leave=False,desc='getting distances')])
    #             for i in trange(len(labels),desc='getting distances',leave=False)]
    distances = []
    for i in trange(len(labels),desc='getting distances',leave=False):
        distances_i = Parallel(n_jobs=-1,verbose=10)(delayed(get_cdf_dist)(scores[sample_indicators[:,i]],
                                                                        result['component_params'],
                                                                        result['weights'][i]) for result in results)
        distances.append(distances_i)
    fig,ax = plt.subplots(1,1)
    sns.boxplot(dict(zip([f"{l} (n={sample_indicators[:,i].sum():,d})" for i,l in enumerate(labels)],distances)),
                orient='h',ax=ax)
    ax.set_xlabel("Distance")
    ax.set_xlim(0,max(np.concatenate(distances))*1.1)
    fig.savefig(save_dir / "cdf_dist.png",bbox_inches='tight',dpi=300)
    plt.close(fig)

def get_out_of_bag_observations(sample_indicators, result):
    bootstrap_indices = result['bootstrap_indices']
    candidate_indices = np.where(sample_indicators.sum(axis=1) > 0)[0]
    training_indices = candidate_indices[bootstrap_indices]
    return np.setdiff1d(np.arange(len(sample_indicators)),training_indices)

def get_oob_preds(scores, sample_indicators, results, Tau_p, Tau_b,point_values,is_inverted,**kwargs):
    oob_preds = defaultdict(list)
    def get_oob_preds_inner(result):
        oob_preds_i = defaultdict(list)
        if sample_indicators.shape[1] > result['weights'].shape[0]:
            SI = sample_indicators[:,:-1]
        else:
            SI = sample_indicators
        oob_indices = get_out_of_bag_observations(SI,result)
        for idx in oob_indices:
            oob_preds_i[idx].append(threshold_utils.assign_assay_evidence_strength(scores[idx],Tau_p, Tau_b, point_values, is_inverted=is_inverted))
        return oob_preds_i
    if kwargs.get("parallel",True):
        oob_pred_dicts = Parallel(n_jobs=-1,verbose=10)(delayed(get_oob_preds_inner)(result) for result in results)
    else:
        oob_pred_dicts = [get_oob_preds_inner(result) for result in tqdm(results)]
    for oob_pred_dict in oob_pred_dicts:
        for idx, preds in oob_pred_dict.items():
            oob_preds[idx].extend(preds)
    tuples = [(k,tuple(v)) for k,v in sorted(list(oob_preds.items()),key=lambda tup: tup[0])]
    indices, oob_preds = list(zip(*tuples))
    return np.array(indices), oob_preds

def get_point_distr_by_class(oob_preds,sample_indicators,labels):
    point_distr = {}
    # labels = list((set(labels) - set(("synonymous",))).union(set(("VUS",))))
    labels_to_include = ["P/LP",'B/LB','gnomAD','VUS']
    include_mask = np.array([label in labels_to_include for label in labels])
    labels = np.array(labels)[include_mask]
    for i,label in enumerate(labels):
        sc = oob_preds[sample_indicators[:,i]]
        if len(sc) == 0:
            continue
        point_distr[label] = {int(points) : int(count) for \
            points, count in zip(*np.unique(sc,return_counts=True))}
    df= pd.DataFrame(point_distr).fillna(0).T.astype(int)
    df.columns = df.columns.astype(int)
    df = df.loc[:,sorted(df.columns)]
    return df

def get_calibration_evidence(scores, hgvs_p,sample_indicators,results, save_dir, Tau_p, Tau_b, point_values, is_inverted,**kwargs):
    threshold_included = np.array([x in point_values for x in np.arange(1,9)])
    TP = Tau_p[threshold_included]
    TB = Tau_b[threshold_included]
    indices, oob_preds = get_oob_preds(scores,sample_indicators,results,TP,TB,point_values,is_inverted,**kwargs)
    oob_pred_mode=np.array([sps.mode(x).mode.item() for x in oob_preds])
    oob_pred_values = [np.unique(v,return_counts=True) for v in oob_preds]
    oob_pred_values = [[(int(k),int(c)) for k,c in zip(*oob)] for oob in oob_pred_values]
    oob_df = pd.DataFrame({'hgvs_p': hgvs_p[np.array(indices)],
                            'scores': scores[np.array(indices)],
                           'oob_pred': oob_pred_mode,
                           'indices': np.array(indices),
                           'sample_indicators': sample_indicators[np.array(indices)].tolist(),
                           'oob_pred_distr': oob_pred_values})
    return oob_df

def generate_oob_fig(scores,hgvs_p,sample_indicators,labels,author_labels,
                        results,save_dir,Tau_p,Tau_b,point_values,is_inverted,
                        scoreset_config,**kwargs):
    rcParams.update(rcParamsDefault)
    oob_df = get_calibration_evidence(scores,
                                        hgvs_p,
                                        sample_indicators,
                                        results,
                                        save_dir,
                                        Tau_p,
                                        Tau_b,
                                        [1,2,4,8],
                                        is_inverted,**kwargs)
    oob_df = oob_df.assign(author_labels=author_labels[np.array(oob_df.indices)])
    oob_df.to_json(save_dir / "oob_df.json")
    point_distr_by_class = get_point_distr_by_class(oob_df.oob_pred,sample_indicators,labels)
    point_distr_by_class.to_csv(save_dir / "point_distr_by_class.csv")
    plt.clf()
    sns.heatmap(point_distr_by_class,cmap='Blues',annot=True,fmt='d',cbar_kws={'label': 'Count'})
    plt.ylabel("Class")
    plt.xlabel("Calibration Points")
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)
    ax.xaxis.label.set_size(22)
    ax.yaxis.label.set_size(20)
    ax.tick_params(axis='y', labelrotation=0)
    fig = plt.gcf()
    fig.savefig(save_dir / f"point_distr_by_label.png",bbox_inches='tight',dpi=300)
    plt.close(fig)

    author_compare_heatmap = pd.pivot_table(oob_df, 
                       index='author_labels',        # Rows (e.g., Date)
                       columns='oob_pred',  # Columns (e.g., Category)
                       values='indices',       # Column to count (can be any column)
                       aggfunc='size',      # Use 'size' to count occurrences
                       fill_value=0)        # Replace NaN with 0
    if not len(author_compare_heatmap):
        return
    author_compare_heatmap.to_csv(save_dir / "point_distr_by_author_class.csv")
    author_compare_heatmap.columns = author_compare_heatmap.columns.astype(int)
    author_compare_heatmap = author_compare_heatmap.loc[:,sorted(author_compare_heatmap.columns)]
    sns.heatmap(author_compare_heatmap,cmap='Blues',annot=True,fmt='d',cbar_kws={'label': 'Count'})
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)
    ax.tick_params(axis='y', labelrotation=0)
    ax.xaxis.label.set_size(22)
    ax.yaxis.label.set_size(20)
    plt.ylabel("Author Classification")
    plt.xlabel("Calibration Points")
    fig = plt.gcf()
    fig.savefig(save_dir / f"point_distr_by_author_class.png",bbox_inches='tight',dpi=300)
    plt.close(fig)

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
    print(f"Generating figs for {dataset_id}")
    save_dir = Path(save_dir) / (dataset_id)
    save_dir.mkdir(exist_ok=True,parents=True)
    print("loading dataset")
    scores, sample_indicators, labels,author_labels,hgvs_p = load_dataset(dataset_id,**kwargs)
    INCLUDES_SYNONYMOUS = "synonymous" in labels
    print("loading results")
    results,result_paths = load_results(dataset_id,results_dirs,includes_synonymous=INCLUDES_SYNONYMOUS,**kwargs)
    if kwargs.get("limitResults",None) is not None:
        results = results[:kwargs["limitResults"]]
        result_paths = result_paths[:kwargs["limitResults"]]
    if not len(results):
        print(f"No results found for {dataset_id}")
        return
    with open(save_dir / "num_results.txt","w") as f:
        f.write(f"{len(results)}")
    print("generating/reloading thresholds")
    tauPFile = save_dir / "Tau_p.npy"
    tauBFile = save_dir / "Tau_b.npy"
    P_file = save_dir / "P.npy"
    B_file = save_dir / "B.npy"
    priors_file = save_dir / "priors.npy"
    is_inverted_file = save_dir / "is_inverted.npy"
    log_lrPlus_file = save_dir / "log_lrPlus.npy"
    if tauPFile.exists() and \
        tauBFile.exists() and \
            P_file.exists() and \
                B_file.exists() and \
                    priors_file.exists() and \
                        is_inverted_file.exists() and \
                            log_lrPlus_file.exists():
        Tau_p = np.load(tauPFile)
        Tau_b = np.load(tauBFile)
        P = np.load(P_file)
        B = np.load(B_file)
        priors = np.load(priors_file)
        is_inverted = np.load(is_inverted_file)
        log_lrPlus = np.load(log_lrPlus_file)
    else:
        print(f"not all files found for {dataset_id}: {tauPFile}, {tauBFile}, {P_file}, {B_file}, {priors_file}, {is_inverted_file}")
        Tau_p, Tau_b, priors,is_inverted,P,B,log_lrPlus = threshold_utils.get_score_threshold(np.linspace(scores.min(),
                                                                                                scores.max(),
                                                                                                3000),
                                                                                    results,**kwargs)
        np.save(save_dir / "Tau_p.npy",Tau_p)
        np.save(save_dir / "Tau_b.npy",Tau_b)
        np.save(save_dir / "P.npy",P)
        np.save(save_dir / "B.npy",B)
        np.save(save_dir / "priors.npy",priors)
        np.save(save_dir / "is_inverted.npy",is_inverted)
        np.save(save_dir / "log_lrPlus.npy",log_lrPlus)
        with open(save_dir / "results_paths.txt","w") as f:
            f.write("\n".join([str(x) for x in result_paths]))
    assert len(Tau_p) == 8
    assert len(Tau_b) == 8
    fig,ax = plt.subplots(1,1)
    rng = np.linspace(scores.min(),scores.max(),3000)
    ax.plot(rng,log_lrPlus.mean(0))
    ax.fill_between(rng,*np.percentile(log_lrPlus,[2.5,97.5],axis=0),alpha=.5)
    ax.set_xlabel("Assay Score")
    ax.set_ylabel("log LR+")
    fig.savefig(save_dir / "log_lrPlus.png",bbox_inches='tight',dpi=300)
    print("prior fig")
    generate_prior_fig(priors,save_dir)
    print("fit fig")
    if labels[-1] == "VUS":
        sL = labels[:-1]
        sI = sample_indicators[:,:-1]
    generate_fit_fig(scores,sL,sI,results,save_dir,Tau_p,Tau_b,priors=priors)
    print("dist fig")
    generate_dist_fig(scores,sL,sI,results,save_dir)
    print("oob fig")
    generate_oob_fig(scores,
                        hgvs_p,
                        sample_indicators,
                        labels,
                        author_labels,
                        results,
                        save_dir,
                        Tau_p,
                        Tau_b,
                        [1,2,4,8],
                        is_inverted,
                        scoreset_config,
                        **kwargs)

def generate_all_figs(processed_data_path,results_dirs,save_dir,**kwargs):
    save_dir = Path(save_dir)
    processed_data_path = Path(processed_data_path)
    config_filepath = kwargs.get("config_filepath","/home/dzeiberg/mave_preprocessing/datasets.yaml")
    with open(config_filepath) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    data_files = list(processed_data_path.glob("*.csv"))
    prioritize_authors = kwargs.get("prioritize_authors",None)
    if prioritize_authors is not None:
        file_order = [x for x in data_files if any([author in str(x.stem) for author in prioritize_authors])]
        file_order = [*file_order, *[x for x in data_files if x not in file_order]]
        print([f.stem for f in file_order])
    for dataset_id in tqdm(file_order,total=len(file_order)):
        font = {'size'   : 24}
        matplotlib.rc('font', **font)
        dataset_id = dataset_id.stem
        dataset_name = dataset_id.split("_pipeline")[0]
        generate_dataset_figs(dataset_id,
                                results_dirs,
                                save_dir,
                                processed_data_path=processed_data_path,
                                scoreset_config=config[dataset_name],
                                **kwargs)

def start(location):
    prioritize_authors = ['Jia']
    if location == "bigticket":
        generate_all_figs(Path("/data/dzeiberg/IGVF-cvfg-pillar-project/Pillar_project_data_files/individual_datasets/"),
                        [Path("/data/dzeiberg/mave_calibration/sc/arion/projects/pejaverlab/users/zeibed01/mave_calibration/results_10_19_24"),],
                        "/data/dzeiberg/mave_calibration/minerva/figs_10_19_24/",
                        config_filepath="/data/dzeiberg/IGVF-cvfg-pillar-project/Pillar_project_data_files/metadata.yaml",
                        prioritize_authors=prioritize_authors)
    else:
        raise ValueError(f"Invalid location: {location}")

if __name__ == "__main__":
    # Fire(start)
    start("bigticket")