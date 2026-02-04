import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path

from utils import generate_filenames, get_algorithm_name

ROOT_PATH = Path(__file__).absolute().resolve().parent


from test_functions import test_ew, test_falcon, test_kliep_mstat


def get_test_function(alg):
    """
    Maps algorithm instances to their corresponding test functions.
    
    Args:
        alg: Algorithm instance
        
    Returns:
        Test function corresponding to the algorithm
    """
    alg_class_name = alg.__class__.__name__
    
    if alg_class_name in ["FS", "FLH"]:
        return test_ew
    elif alg_class_name == "FALCON":
        return test_falcon
    elif alg_class_name in ["KLIEP", "Mstatistic"]:
        return test_kliep_mstat
    else:
        raise ValueError(f"Unknown algorithm class: {alg_class_name}. "
                        f"Supported algorithms: FS, FLH, FALCON, KLIEP, Mstatistic")

@hydra.main(version_base=None, config_path="configs", config_name="run")
def main(config: DictConfig):
    mode = config.runner.mode
    dataset = instantiate(config.data)
    alg = instantiate(config.algorithm)

    test_func = get_test_function(alg)
    
    dataset_name = config.runner.dataset_name
    custom_names = None
    if hasattr(config.runner, 'output_names'):
        custom_names = config.runner.output_names
    
    
    filenames = generate_filenames(alg, dataset_name, mode, 
                                   config.runner.save_path, custom_names)
    
    
    kwargs = config.kwargs if hasattr(config, 'kwargs') else {}
    alg_name = get_algorithm_name(alg)
    if alg_name in kwargs:
        kwargs = dict(kwargs[alg_name])
    else:
        kwargs = {}

    kwargs["dataset_name"] = dataset_name
    if hasattr(config.runner, 'thresholds_csv_path'):
        kwargs['thresholds_csv_path'] = config.runner.thresholds_csv_path

    if hasattr(config.runner, 'threshold_scaling_path'):
        kwargs['threshold_scaling_path'] = config.runner.threshold_scaling_path

    if mode == "changepoint":
        kwargs['plot_path'] = filenames['plot']
    
    
    results = test_func(alg, mode, dataset, **kwargs)
    results = pd.DataFrame(results)
    results.to_csv(filenames['csv'], index=False)
    
    print(f"Results saved to: {filenames['csv']}")
    if mode == "changepoint":
        print(f"Plot saved to: {filenames['plot']}")


if __name__ == "__main__": 
    main()
