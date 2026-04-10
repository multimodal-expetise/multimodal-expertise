import torch
import torch.nn as nn
import os
from tqdm import tqdm
from config.config import get_config_regression
from data_loader import MMDataLoader
from models import AMIO
from utils import *


def _resolve_checkpoint_path(model_name):
    state_dict_path = fr"pretrained_model/{model_name}_pretrained_model_state_dict.pth"
    legacy_path = fr"pretrained_model/{model_name}_pretrained_model.pth"
    return state_dict_path if os.path.exists(state_dict_path) else legacy_path


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if "model" in checkpoint:
            raise RuntimeError(
                "Checkpoint contains serialized model object; convert it with convert_legacy_checkpoint.py first."
            )
        return checkpoint
    raise RuntimeError("Unsupported checkpoint format. Expecting state_dict checkpoint.")


def _infer_arch_hints_from_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    hints = {}
    proj_key = "Model.proj_l.weight"
    trans_key = "Model.trans.layers.0.self_attn.in_proj_weight"

    if proj_key in state_dict:
        hints["dst_feature_dim"] = int(state_dict[proj_key].shape[0])

    if proj_key in state_dict and trans_key in state_dict:
        proj_dim = int(state_dict[proj_key].shape[0])
        trans_embed_dim = int(state_dict[trans_key].shape[1])
        hints["legacy_early_concat_feature"] = (trans_embed_dim == proj_dim * 3)

    return hints


def load_pretrained_weights(model, model_path, device):
    """Load state_dict checkpoint for strict reproducibility."""
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint: {model_path}\n"
            "This evaluation script only accepts state_dict checkpoints.\n"
            "Please convert legacy object checkpoints first, e.g.:\n"
            "python convert_legacy_checkpoint.py <old_checkpoint_path>"
        ) from e

    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    return model

def test(model_name, dataset_name, featurePath):
    """Main testing function for multimodal regression model.

    Args:
        model_name (str): Name of the model architecture
        dataset_name (str): Name of the dataset
        featurePath (str): Path to preprocessed feature file
    """
    # Set a fixed seed for reproducibility
    set_seed(42)

    # Load configuration from JSON file
    config_file = r"config/config_pretrained.json"
    args = get_config_regression(model_name, dataset_name, config_file)

    # Update arguments with feature path and device selection
    args["featurePath"] = featurePath
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained weights
    model_path = _resolve_checkpoint_path(model_name)


    # Align model hidden size with checkpoint if needed (e.g., 30 vs 36).
    hints = _infer_arch_hints_from_checkpoint(model_path, args["device"])
    ckpt_dim = hints.get("dst_feature_dim")
    if ckpt_dim is not None and "dst_feature_dim_nheads" in args:
        current = list(args["dst_feature_dim_nheads"])
        if current[0] != ckpt_dim:
            current[0] = ckpt_dim
            args["dst_feature_dim_nheads"] = current

    if hints.get("legacy_early_concat_feature", False):
        args["legacy_early_concat_feature"] = True


    # Initialize data loader with batch size 1
    dataloader = MMDataLoader(args, 1)

    # Initialize model and move to appropriate device
    model = AMIO(args).to(args['device'])

    model = load_pretrained_weights(model, model_path, args['device'])
    model.to(args['device'])

    # Execute testing procedure
    do_test(args, model_name, dataset_name, model, dataloader)


def do_test(args, model_name, dataset_name, model, dataloader):

    criterion = nn.L1Loss()

    metrics = MetricsTop(args['test_mode']).getMetrics(args['dataset_name'])

    model.eval()

    y_pred, y_true = [], []
    info_s = []

    eval_loss = 0.0

    with torch.no_grad():

        for batch_data in tqdm(dataloader, desc="Processing", ncols=100):

            vision = batch_data['vision'].to(args['device'])
            audio = batch_data['audio'].to(args['device'])
            text = batch_data['text'].to(args['device'])

            labels = batch_data['labels']['M'].to(args['device'])
            labels = labels.view(-1, 1)

            info = batch_data.get("meta_info", None)
            if info is not None:
                info_s += info

            outputs = model(text, audio, vision)

            if isinstance(outputs, dict):
                outputs = outputs['M']

            loss = criterion(outputs, labels)
            eval_loss += loss.item()

            y_pred.append(outputs.cpu())
            y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)

        pred = torch.cat(y_pred)
        true = torch.cat(y_true)

    eval_results = metrics(pred, true)

    print("The results are as follows:")
    print(eval_results)
