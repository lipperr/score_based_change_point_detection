

def get_algorithm_name(alg):
    """
    Get a short name for the algorithm for use in filenames.
    
    Args:
        alg: Algorithm instance
        
    Returns:
        Short algorithm name (lowercase)
    """
    if isinstance(alg, str):
        return alg
    
    alg_class_name = alg.__class__.__name__
    name_mapping = {
        "FS": "fs",
        "FLH": "flh",
        "FALCON": "falcon",
        "KLIEP": "kliep",
        "Mstatistic": "mstat"
    }
    return name_mapping.get(alg_class_name, alg_class_name.lower())

def generate_filenames(alg, dataset_name, mode, save_path, custom_names=None):
    """
    Generate filenames for CSV and plot outputs.
    
    Args:
        alg: Algorithm instance
        dataset_name: Name of the dataset
        mode: 'stationary' or 'changepoint'
        save_path: Base path for saving files
        custom_names: Optional dict with 'csv' and/or 'plot' keys for custom names
        
    Returns:
        Dict with 'csv' and 'plot' keys containing full file paths
    """
    alg_name = get_algorithm_name(alg)
    
    csv_name = f"{mode}_{alg_name}_{dataset_name}_ths.csv"
    plot_name = f"{mode}_{alg_name}_{dataset_name}.png"
    
    if custom_names:
        if 'csv' in custom_names:
            csv_name = custom_names['csv']
        if 'plot' in custom_names:
            plot_name = custom_names['plot']
    
    if not save_path.endswith('/'):
        save_path += '/'
    
    return {
        'csv': save_path + csv_name,
        'plot': save_path + plot_name
    }

