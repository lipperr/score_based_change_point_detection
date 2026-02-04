import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path

from utils import generate_filenames

ROOT_PATH = Path(__file__).absolute().resolve().parent



@hydra.main(version_base=None, config_path="configs", config_name="run")
def main(config: DictConfig):   
    
    dataset_name = config.runner.dataset_name
    custom_names = None
    if hasattr(config.runner, 'output_names'):
        custom_names = config.runner.output_names
    
    filenames = generate_filenames(alg="", dataset_name=dataset_name, mode = "",
                                   save_path=config.runner.save_path, custom_names=custom_names)
    dataset = instantiate(config.data)
    dataset.display_data(filenames['plot'])

    print(f"Plot saved to: {filenames['plot']}")


if __name__ == "__main__": 
    main()
