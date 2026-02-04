import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
from utils import generate_filenames
from test_functions import test_all

ROOT_PATH = Path(__file__).absolute().resolve().parent

@hydra.main(version_base=None, config_path="configs", config_name="run_for_plot")
def main(config: DictConfig):

    mode = config.runner.mode
    dataset = instantiate(config.data)
    instances = {}
    for key, instance_cfg in config.instances.items():
        if "_target_" in instance_cfg:
            instances[key] = instantiate(instance_cfg)

    dataset_name = config.runner.dataset_name
    custom_names = None
    if hasattr(config.runner, 'output_names'):
        custom_names = config.runner.output_names
    
    
    filenames = generate_filenames("all", dataset_name, mode,
                                   config.runner.save_path, custom_names)
    
    
    kwargs = config.kwargs if hasattr(config, 'kwargs') else {}

    if hasattr(config.runner, 'thresholds_csv_path'):
        kwargs['thresholds_csv_path'] = config.runner.thresholds_csv_path
        
    if hasattr(config.runner, 'threshold_scale'):
        kwargs['threshold_scale'] = config.runner.threshold_scale
    
    if mode == "changepoint":
        kwargs['plot_path'] = filenames['plot']
    
    
    results = test_all(instances, mode, dataset, **kwargs)
    results = pd.DataFrame(results)
    results.to_csv(filenames['csv'], index=False)
    
    print(f"Results saved to: {filenames['csv']}")
    if mode == "changepoint":
        print(f"Plot saved to: {filenames['plot']}")

if __name__ == "__main__": 
    main()


    