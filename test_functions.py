import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
from typing import Dict, Tuple, Any, Optional, List

from utils import get_algorithm_name
from run_procedure import tune_threshold, run_changepoint, performance_run

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

param_columns = {"ew": ["alpha", "lambda", "eta", "gamma"], 
                 "falcon": ["p", "beta"], 
                 "kliep_mstat": ["window_size", "sigma"]}

name_mapping = {
        "fs": "Score-based",
        "flh": "FLH",
        "falcon": "FALCON",
        "kliep": "KLIEP",
        "mstat": "M-statistic"
    }
    
def _load_threshold_scaling_config(path: Optional[str]) -> Dict:
    """Load threshold scaling config from YAML. Returns {} if path is None or file missing."""
    if path is None or OmegaConf is None:
        return {}
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / path
    if not p.exists():
        return {}
    try:
        cfg = OmegaConf.load(p)
        return OmegaConf.to_container(cfg, resolve=True) if cfg else {}
    except Exception:
        return {}


def apply_threshold_scaling(
    threshold: float,
    alg_name: str,
    dataset_name: str,
    scaling_config: Dict,
) -> float:
    """
    Apply optional scaling to a tuned threshold from config.
    Config is nested: scaling_config[dataset_name][alg_name] -> { factor?, add? }.
    Applied as: threshold = threshold * factor + add; 
    """
    if not scaling_config:
        print("Couldn't find scaling config.")
        return threshold
    by_dataset = scaling_config.get(dataset_name) or {}
    rules = by_dataset.get(alg_name) if isinstance(by_dataset, dict) else None
    if not rules or not isinstance(rules, dict):
        print("Couldn't find a scaling rule.")
        return threshold
    out = threshold
    if "factor" in rules:
        out = out * float(rules["factor"])
    if "add" in rules:
        out = out + float(rules["add"])
    return out



def check_existing_thresholds(
    thresholds_csv_path: str,
    param_columns: Optional[List[str]] = None,
    threshold_column: str = "threshold",
    verbose: bool = True
) -> Dict[Tuple[Any, ...], float]:
    """
    Universal function to load precomputed thresholds from a CSV file.
    
    Args:
        thresholds_csv_path: Path to the CSV file containing thresholds
        param_columns: List of parameter column names. If None, will try to 
                      infer parameters (all columns except threshold_column)
        threshold_column: Name of the column containing threshold values
        verbose: Whether to print status messages
    
    Returns:
        Dictionary mapping parameter tuples to threshold values
    """
    thresholds_dict = {}
    
    if verbose:
        print(f"Searching for precomputed thresholds in {thresholds_csv_path}...")
    
    try:
        df = pd.read_csv(thresholds_csv_path)
        
        if param_columns is None:
            param_columns = [col for col in df.columns if col != threshold_column]
        
        missing_cols = [col for col in param_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing parameter columns in CSV: {missing_cols}")
        
        if threshold_column not in df.columns:
            raise ValueError(f"Missing threshold column '{threshold_column}' in CSV")
        
        for _, row in df.iterrows():
            param_values = []
            for param in param_columns:
                val = row[param]
                try:
                    if pd.isna(val):
                        param_values.append(val)
                    else:
                        param_values.append(float(val))
                except (ValueError, TypeError):
                    param_values.append(val)
            
            threshold_val = float(row[threshold_column])
            
            thresholds_dict[tuple(param_values)] = threshold_val
        
        if verbose:
            print(f"Loaded {len(thresholds_dict)} thresholds from {thresholds_csv_path}")
            if param_columns:
                print(f"Parameters used: {param_columns}")
    
    except FileNotFoundError:
        if verbose:
            print(f"Warning: Thresholds file not found at {thresholds_csv_path}")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not load thresholds from CSV: {e}")
    
    return thresholds_dict


def test_ew(alg, mode, dataset, **kwargs): 
        params = param_columns["ew"]
        results = dict()
        for param in params:
            results[param] = []
        results["threshold"] = []

        alpha_list = kwargs.get("alpha_list", [alg.alpha])
        lambda_list = kwargs.get("lambda_list", [alg.lambda_])
        eta_list = kwargs.get("eta_list", [alg.eta]) 
        gamma_list = kwargs.get("gamma_list", [alg.gamma]) 
        thresholds_csv_path = kwargs.get("thresholds_csv_path", None)
        
        data_stat = dataset.get_data_stat()
        thresholds_dict = {}
        if mode == "stationary":
            for a in alpha_list:
                for l in lambda_list:
                    for e in eta_list:
                        for g in gamma_list:
                            alg.alpha, alg.lambda_, alg.eta, alg.gamma = a, l, e, g
                            threshold = tune_threshold(alg, data_stat)
                            results["alpha"].append(a)
                            results["lambda"].append(l)
                            results["eta"].append(e)
                            results["gamma"].append(g)
                            results["threshold"].append(threshold)

        elif mode == "changepoint":
            if  thresholds_csv_path and os.path.exists(thresholds_csv_path):
                thresholds_dict = check_existing_thresholds(thresholds_csv_path, param_columns=param_columns["ew"])
            scaling_path = kwargs.get("threshold_scaling_path", "configs/threshold_scaling.yaml")
            scaling_config = _load_threshold_scaling_config(scaling_path)
            alg_name = get_algorithm_name(alg)
            plot_path = kwargs.get("plot_path", "results/output.png")
            plt.figure(figsize=(10, 5))
            results["FA"] = []
            results["DD"] = []
            results["ND"] = []
            for a in alpha_list:
                for l in lambda_list:
                    for e in eta_list:
                        for g in gamma_list:
                            alg.alpha, alg.lambda_, alg.eta, alg.gamma = a, l, e, g
                            
                            if (a, l, e, g) in thresholds_dict:
                                threshold = thresholds_dict[(a, l, e, g)]
                                print(f"Using precomputed threshold for alpha={a}, lambda={l}, eta={e}, gamma={g}, ths={np.round(threshold, 3)}")
                            else:
                                threshold = tune_threshold(alg, data_stat)
                                print(f"Computed threshold for alpha={a}, lambda={l}, eta={e}, gamma={g}, ths={np.round(threshold, 3)}")
                            threshold = apply_threshold_scaling(
                                threshold, alg_name, getattr(dataset, "name", ""), scaling_config
                            )

                            S, return_data = run_changepoint(alg, dataset, threshold)
                            for i in range(len(return_data)):
                                fa, ddm, dds, nd, thshd = return_data[i]
                                results["alpha"].append(a)
                                results["lambda"].append(l)
                                results["eta"].append(e)
                                results["gamma"].append(g)
                                results["threshold"].append(thshd)
                                results["FA"].append(fa)
                                results["DD"].append(f"{ddm}$\pm${dds}")
                                results["ND"].append(nd)
                                
                                print(f"a={a}, l={l}, e={e}, g={g}, ths: {np.round(thshd, 3)}, fa: {fa}, dd: {ddm}$\pm${dds}, nd: {nd}")
                            plt.plot(S, label=f"{a}, {l}, {e}, {g}")

            cps = dataset.get_cps_plot()
            for cp in cps:
                plt.axvline(cp, linestyle=":")
            plt.legend()
            os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else ".", exist_ok=True)
            plt.savefig(plot_path)
            plt.close()

        else:
            raise RuntimeError("Invalid running mode")
        return results


def test_falcon(alg, mode, dataset, **kwargs):
        p_list = kwargs.get("p_list", [alg.p])
        beta_list = kwargs.get("beta_list", [alg.beta])
        thresholds_csv_path = kwargs.get("thresholds_csv_path", None)
        
        params = param_columns["falcon"]
        results = dict()
        for param in params:
            results[param] = []
        results["threshold"] = []
        
        data_stat = dataset.get_data_stat()
        thresholds_dict = {}

        if mode == "stationary":
            for p in p_list:
                for beta in beta_list:
                    alg.p, alg.beta = p, beta
                    threshold = tune_threshold(alg, data_stat)
                    results["p"].append(p)
                    results["beta"].append(beta)
                    results["threshold"].append(threshold)
                    print(f"{p}, {beta}, {threshold}")

        elif mode == "changepoint":
            if thresholds_csv_path and os.path.exists(thresholds_csv_path):
                thresholds_dict = check_existing_thresholds(thresholds_csv_path, param_columns=param_columns["falcon"])
            scaling_path = kwargs.get("threshold_scaling_path", "configs/threshold_scaling.yaml")
            scaling_config = _load_threshold_scaling_config(scaling_path)
            alg_name = get_algorithm_name(alg)
            plot_path = kwargs.get("plot_path", "results/falcon_1dmean.png")
            plt.figure(figsize=(10, 5))
            results["FA"] = []
            results["DD"] = []
            results["ND"] = []
            
            for p in p_list:
                for beta in beta_list:

                    alg.p, alg.beta = p, beta
                    if (p, beta) in thresholds_dict:
                        threshold = thresholds_dict[(p, beta)]
                        print(f"Using precomputed threshold for p={p}, beta={beta}, ths={np.round(threshold, 3)}")
                    else:
                        threshold = tune_threshold(alg, data_stat)
                        print(f"Computed threshold for p={p}, beta={beta}, ths={np.round(threshold, 3)}")
                    threshold = apply_threshold_scaling(
                        threshold, alg_name, getattr(dataset, "name", ""), scaling_config
                    )

                    S, return_data = run_changepoint(alg, dataset, threshold=threshold)
                    for i in range(len(return_data)):
                        fa, ddm, dds, nd, thshd = return_data[i]
                        results["p"].append(p)
                        results["beta"].append(beta)
                        results["threshold"].append(thshd)
                        results["FA"].append(fa)
                        results["DD"].append(f"{ddm}$\pm${dds}")
                        results["ND"].append(nd)
                        print(f"p={p}, beta={beta}, ths: {np.round(thshd, 3)}, fa: {fa}, dd: {ddm}$\pm${dds}, nd: {nd}")
                    plt.plot(S, label=f"{p}, {beta}")

            cps = dataset.get_cps_plot()
            for cp in cps:
                plt.axvline(cp, linestyle=":")
            plt.legend()
            os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else ".", exist_ok=True)
            plt.savefig(plot_path)
            plt.close()
        else:
            raise RuntimeError("Invalid running mode")
        return results

def test_kliep_mstat(alg, mode, dataset, **kwargs):
        params = param_columns["kliep_mstat"]
        results = dict()
        for param in params:
            results[param] = []
        results["threshold"] = []
        
        default_sigma = getattr(alg, 'sigma', 1)
        default_ws = getattr(alg, 'window_size', 30)
        sigma_list = kwargs.get("sigma_list", [default_sigma])
        window_size_list = kwargs.get("window_size_list", [default_ws])
        thresholds_csv_path = kwargs.get("thresholds_csv_path", None)
        
        data_stat = dataset.get_data_stat()
        thresholds_dict = {}
        
        if mode == "stationary":
            for ws in window_size_list:
                for sigma in sigma_list:
                    alg.window_size, alg.sigma = ws, sigma        
                    threshold = tune_threshold(alg, data_stat)
                    results["sigma"].append(sigma)
                    results["window_size"].append(ws)
                    results["threshold"].append(threshold)

        elif mode == "changepoint":
            if thresholds_csv_path and os.path.exists(thresholds_csv_path):
                thresholds_dict = check_existing_thresholds(thresholds_csv_path, param_columns=param_columns["kliep_mstat"])
            scaling_path = kwargs.get("threshold_scaling_path", "configs/threshold_scaling.yaml")
            scaling_config = _load_threshold_scaling_config(scaling_path)
            alg_name = get_algorithm_name(alg)
            plot_path = kwargs.get("plot_path", "results/kliep_mstat_1dmean.png")
            plt.figure(figsize=(10, 5))
            results["FA"] = []
            results["DD"] = []
            results["ND"] = []
            for sigma in sigma_list:
                for ws in window_size_list:
                    alg.window_size, alg.sigma = ws, sigma

                    if (ws, sigma) in thresholds_dict:
                        threshold = thresholds_dict[(ws, sigma)]
                        print(f"Using precomputed threshold for b={sigma}, ths={np.round(threshold, 3)}")
                    else:
                        threshold = tune_threshold(alg, data_stat)
                        print(f"Computed threshold for sigma={sigma}, ths={np.round(threshold, 3)}")

                    threshold = apply_threshold_scaling(
                        threshold, alg_name, getattr(dataset, "name", ""), scaling_config
                    )
                    S, return_data = run_changepoint(alg, dataset, threshold)
                    for i in range(len(return_data)):
                        fa, ddm, dds, nd, thshd = return_data[i]
                        results["sigma"].append(sigma)
                        results["window_size"].append(ws)
                        results["threshold"].append(thshd)
                        results["FA"].append(fa)
                        results["DD"].append(f"{ddm}$\pm${dds}")
                        results["ND"].append(nd)
                        print(f"sigma={sigma}, ws={ws}, ths: {np.round(thshd, 3)}, fa: {fa}, dd: {ddm}$\pm${dds}, nd: {nd}")
                    plt.plot(S, label=f"{sigma}")
            

            cps = dataset.get_cps_plot()
            for cp in cps:
                plt.axvline(cp, linestyle=":")
            plt.legend()
            os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else ".", exist_ok=True)
            plt.savefig(plot_path)
            plt.close()

        else:
            raise RuntimeError("Invalid running mode")
        return results


def test_all(instances, mode, dataset, **kwargs):
    c = {'fs': 'r', 'flh': 'b', 'falcon': 'g', 'kliep': 'm', 'mstat': 'c'}

    thresholds_csv_path = kwargs.get("thresholds_csv_path", None)
    results = dict()
    results["threshold"] = []
    
    data_stat = dataset.get_data_stat()
    
    thresholds_dict = {}
    if mode == "changepoint" and thresholds_csv_path and os.path.exists(thresholds_csv_path):
        print("Searching for precomputed thresholds...")
        try:
            df = pd.read_csv(thresholds_csv_path)
            for alg_name in instances.keys():
                if alg_name in df.columns():
                    thresholds_dict[alg_name] = df[alg_name]
                else:
                    raise ValueError(f"CSV must contain {alg_name} column")
            print(f"Loaded {len(thresholds_dict)} thresholds from {thresholds_csv_path}")
        except Exception as e:
            print(f"Warning: Could not load thresholds from CSV: {e}")
            thresholds_dict = {}

    if mode == "changepoint":
        data = dataset.get_data_cp()
        cps = dataset.get_cps()

        plot_path = kwargs.get("plot_path", "results/3dmean.png")
        fig, ax = plt.subplots(2, 1, figsize=(15, 7))
        ax[0].set_title('Time series', fontsize=24)
        ax[0].set_xlabel('Time', fontsize='16')
        ax[0].set_ylabel('Signal', fontsize='16')
        ax[0].plot(data[0])
        ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        results["FA"] = []
        results["DD"] = []
        results["ND"] = []
        results["stopping_time"] = []

        for alg_name, alg in instances.items():

            if alg_name in thresholds_dict:
                threshold = thresholds_dict[alg_name]
                print(f"Using precomputed threshold = {np.round(threshold, 3)} for {alg_name}")
            elif not np.isinf(alg.threshold):
                threshold = alg.threshold
                print(f"Using manually set threshold = {np.round(threshold, 3)} for {alg_name}")
            else:
                threshold = tune_threshold(alg, data_stat)
                print(f"Computed threshold = {np.round(threshold, 3)} for {alg_name}")

            S, return_data = performance_run(alg, dataset, threshold)

            st, fa, delay, nd, thshd = return_data
            results["stopping_time"].append(st)
            results["threshold"].append(thshd)
            results["FA"].append(fa)
            results["DD"].append(delay)
            results["ND"].append(nd)

            S /= S.max()
            ax[1].plot(S, label=name_mapping.get(alg_name, alg_name), color=c[alg_name])
            ax[1].plot([st], [S[st]], 'o', color=c[alg_name] )
            
        ax[1].set_title('Test statistic', fontsize=24)
        ax[1].set_xlabel('Time', fontsize='16')
        ax[1].set_ylabel('Statistic', fontsize='16')
        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax[0].axvline(cps[0], c='black', ls=':', label='True change point')
        ax[1].axvline(cps[0], c='black', ls=':', label='True change point')

        plt.tight_layout()
        plt.legend()
        os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else ".", exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
    else:
        raise RuntimeError("Invalid running mode")

    return results

