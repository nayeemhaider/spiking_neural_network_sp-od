# test.py (Modified)

"""
Runs inference and evaluation on a pre-trained radar perception model.

This script loads a model checkpoint, automatically determines the model
architecture (CNN, Hybrid SNN, or End-to-End SNN) from the experiment name,
runs it on a specified test dataset, and generates evaluation plots and metrics.
"""

import os
import torch
import json
import argparse
from tqdm import tqdm

# Import all possible models and configurations
from src.config import TRAINING_CONFIG, SP_CONFIG, RADAR_CONFIG
from src.model import UnifiedRadarSNN_Hybrid, UnifiedRadarSNN_End2End
from src.cnn_model import UnifiedRadarCNN
from src.data_loader import UnifiedRadarDataset, load_real_radar_data
from src.utils import (setup_logging, evaluate_and_visualize_sp, evaluate_and_visualize_od,
                       plot_od_predictions_on_rdm)
from torch.utils.data import DataLoader


def run_inference(model, data_loader, device):
    """
    Runs the model in evaluation mode on the provided data and collects all outputs.
    """
    model.eval()
    all_sp_preds, all_sp_targets, all_od_preds, all_od_targets = [], [], [], []
    
    progress_bar = tqdm(data_loader, desc="Running Inference on Test Data")
    
    with torch.no_grad():
        for batch in progress_bar:
            if not batch or len(batch) < 3: continue
            snn_input, sp_targets, od_targets = batch
            if snn_input.nelement() == 0: continue

            snn_input = snn_input.to(device)
            predictions = model(snn_input)

            all_sp_preds.append({k: v.cpu().detach() for k, v in predictions['sp_outputs'].items()})
            all_sp_targets.append({k: v.cpu().detach() for k, v in sp_targets.items()})
            all_od_preds.append({k: v.cpu().detach() for k, v in predictions['od_outputs'].items()})
            all_od_targets.append({k: v.cpu().detach() for k, v in od_targets.items()})

    if not all_sp_preds:
        print("Warning: No valid data was processed during inference.")
        return None, None, None, None

    final_sp_preds = {k: torch.cat([p[k] for p in all_sp_preds]) for k in all_sp_preds[0]}
    final_sp_targets = {k: torch.cat([t[k] for t in all_sp_targets]) for k in all_sp_targets[0]}
    final_od_preds = {k: torch.cat([p[k] for p in all_od_preds]) for k in all_od_preds[0]}
    final_od_targets = {k: torch.cat([t[k] for t in all_od_targets]) for k in all_od_targets[0]}

    return final_sp_preds, final_sp_targets, final_od_preds, final_od_targets


def main(args):
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return

    model_dir = os.path.dirname(args.model_path)
    test_results_dir = os.path.join(model_dir, f"test_evaluation_{os.path.basename(args.model_path).split('.')[0]}")
    os.makedirs(test_results_dir, exist_ok=True)
    
    logger = setup_logging(test_results_dir)
    logger.info("--- Starting Test Evaluation ---")
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Test Data Path: {args.data_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- Load Model Configuration from the experiment's JSON file ---
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        logger.error(f"FATAL: 'config.json' not found in '{model_dir}'. Cannot determine model parameters.")
        return
        
    with open(config_path, 'r') as f:
        saved_configs = json.load(f)
    params = saved_configs # The entire dictionary of parameters

    # --- Determine Model Architecture from directory name ---
    dir_name = os.path.basename(model_dir).lower()
    if 'snn_hybrid' in dir_name:
        model_class = UnifiedRadarSNN_Hybrid
    elif 'snn_e2e' in dir_name:
        model_class = UnifiedRadarSNN_End2End
    elif 'cnn' in dir_name:
        model_class = UnifiedRadarCNN
    else:
        logger.error(f"Could not determine model type from directory name: {dir_name}. Please ensure directory name contains 'cnn', 'snn_hybrid', or 'snn_e2e'.")
        return
    logger.info(f"Inferred model type: {model_class.__name__}")

    # --- Initialize and Load Model ---
    model = model_class(**params).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    logger.info("Successfully loaded model weights.")

    # --- Load Test Data ---
    sequence_names = args.sequences.split(',')
    logger.info(f"Loading test data for sequences: {sequence_names}")
    radar_data, labels = load_real_radar_data(
        args.data_path, sequence_names, max_frames_per_seq=args.max_frames
    )
    if len(radar_data) == 0:
        logger.error("No data files found. Please check data path and sequence names.")
        return

    test_dataset = UnifiedRadarDataset(
        radar_data, labels,
        time_steps=params.get('time_steps', 128),
        sp_config=params,
        train_config=params
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Run Inference and Evaluate ---
    results = run_inference(model, test_loader, device)
    if results is None:
        logger.error("Inference failed.")
        return
        
    sp_preds, sp_targets, od_preds, od_targets = results
    logger.info("Inference complete. Generating evaluation plots and reports...")
    
    evaluate_and_visualize_sp([sp_preds], [sp_targets], logger, test_results_dir)
    evaluate_and_visualize_od(od_preds, od_targets, logger, test_results_dir)
    plot_od_predictions_on_rdm(sp_preds, od_preds, od_targets, params, logger, test_results_dir)
    
    logger.info(f"--- Test Evaluation Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run evaluation on a trained radar model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file (.pth).")
    parser.add_argument('--data_path', type=str, default=r"D:\Automotive", help="Path to the root of the dataset directory.")
    parser.add_argument('--sequences', type=str, required=True, help="Comma-separated list of sequence folder names for testing.")
    parser.add_argument('--max_frames', type=int, default=None, help="Maximum number of frames to load per sequence (optional).")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for inference.")
    
    main(parser.parse_args())

