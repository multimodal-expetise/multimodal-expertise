import json
from pathlib import Path
from easydict import EasyDict as edict


def get_config_regression(model_name, dataset_name, config_file=""):
    if config_file == "":
        config_file = Path(__file__).parent / "config_pretrained.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)

    if model_name not in config_all:
        raise KeyError(f"Model '{model_name}' not found in {config_file}. "
                       f"Available: {[k for k in config_all if k.startswith('A')]}")
    if dataset_name not in config_all.get('datasetCommonParams', {}):
        raise KeyError(f"Dataset '{dataset_name}' not found in {config_file}.")

    model_dataset_args = config_all[model_name]
    dataset_args = config_all['datasetCommonParams'][dataset_name]

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config['test_mode'] = 'regression'
    config.update(dataset_args)
    config.update(model_dataset_args)
    config = edict(config)

    return config
