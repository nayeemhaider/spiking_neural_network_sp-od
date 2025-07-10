"""
Main script to run a training and evaluation experiment for various radar models.

This script serves as the primary entry point for training and evaluating
different architectures on the unified radar perception task. It supports:
- 'cnn': A baseline CNN+GRU model.
- 'cnn_mlp': A simpler CNN+MLP model without a recurrent component.
- 'snn_hybrid': A hybrid SNN with a GRU decoder and analog output heads.
- 'snn_e2e': A fully end-to-end spiking SNN using rate-coding.

It handles experiment setup, data loading, model initialization, training,
and comprehensive final evaluation on the test set.
"""
import os
import datetime
import torch
import json
import logging
import argparse
from src.model import UnifiedRadarSNN_Hybrid, UnifiedRadarSNN_End2End
from src.cnn_model import UnifiedRadarCNN, UnifiedRadarCNN_MLP
from src.config import TRAINING_CONFIG, SP_CONFIG, RADAR_CONFIG, BASE_DATA_PATH, RESULT_BASE_DIR
from src.data_loader import create_data_loaders
from src.trainer import Trainer
from src.utils import (
    setup_logging, save_config, plot_training_curves, plot_batch_losses,
    evaluate_and_visualize_sp, evaluate_and_visualize_od,
    plot_od_predictions_on_rdm
)


def main(args):
    """Orchestrates the full experiment pipeline."""

    # --- 1. SETUP EXPERIMENT ---
    model_map = {
        'cnn': (UnifiedRadarCNN, "CNN_GRU_Baseline"),
        'cnn_mlp': (UnifiedRadarCNN_MLP, "CNN_MLP_Baseline"),
        'snn_hybrid': (UnifiedRadarSNN_Hybrid, "SNN_Hybrid_GRU"),
        'snn_e2e': (UnifiedRadarSNN_End2End, "SNN_End2End")
    }

    # Handle 'snn' as an alias for 'snn_hybrid' for backward compatibility
    model_choice = 'snn_hybrid' if args.model_type == 'snn' else args.model_type
    model_class, name_prefix = model_map[model_choice]

    # Create a unique directory for this experiment's results
    exp_name = f"{name_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_dir = os.path.join(RESULT_BASE_DIR, exp_name)
    os.makedirs(result_dir, exist_ok=True)
    
    logger = setup_logging(result_dir)

    # --- 2. CONFIGURE AND INITIALIZE ---
    # Combine all configurations into a single parameter dictionary
    params = {**RADAR_CONFIG, **SP_CONFIG, **TRAINING_CONFIG}
    params['dataset_path'] = BASE_DATA_PATH  # Use path from config
    params['num_samples'] = params['numAdcSamples'] # Ensure key consistency

    device = torch.device(params['device'])
    model = model_class(**params).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- 3. LOAD DATA ---
    try:
        train_loader, val_loader, test_loader = create_data_loaders(params, params)
    except FileNotFoundError as e:
        logger.error(f"CRITICAL DATA LOADING ERROR: {e}")
        logger.error(f"Please ensure the dataset is available at the specified path: {params['dataset_path']}")
        return

    # --- 4. LOG EXPERIMENT SUMMARY ---
    logger.info("="*70)
    logger.info(f" Starting Experiment: {exp_name}")
    logger.info("="*70)
    logger.info(f"  - Model Architecture:     {model_class.__name__}")
    logger.info(f"  - Execution Device:       {str(device).upper()}")
    logger.info(f"  - Trainable Parameters:   {total_params:,}")
    logger.info("-" * 70)
    logger.info("  TRAINING CONFIGURATION:")
    logger.info(f"  - Learning Rate:          {params['learning_rate']}")
    logger.info(f"  - Epochs:                 {params['num_epochs']}")
    logger.info(f"  - Batch Size:             {params['batch_size']}")
    logger.info(f"  - Time Steps (Chirps):    {params['time_steps']}")
    logger.info("-" * 70)
    logger.info("  DATASET:")
    logger.info(f"  - Training samples:       {len(train_loader.dataset)}")
    logger.info(f"  - Validation samples:     {len(val_loader.dataset)}")
    logger.info(f"  - Test samples:           {len(test_loader.dataset)}")
    logger.info("="*70)

    # Save the final configuration for full reproducibility
    save_config({
        'model_class': model_class.__name__, 'total_params': total_params, 'args': vars(args),
        'training_config': TRAINING_CONFIG, 'sp_config': SP_CONFIG, 'radar_config': RADAR_CONFIG
    }, result_dir)


    # --- 5. RUN TRAINING AND EVALUATION ---
    try:
        # Train the model
        trainer = Trainer(model, train_loader, val_loader, params, result_dir)
        history = trainer.train(logger)
        plot_training_curves(history, result_dir)
        plot_batch_losses(result_dir)
        logger.info("Training complete. All training curves have been generated.")

        # Evaluate on the test set
        logger.info("--- Starting Final Evaluation on Test Set ---")
        best_model_path = os.path.join(result_dir, 'best_model.pth')
        if not os.path.exists(best_model_path):
            logger.warning("Best model checkpoint not found. Skipping final evaluation.")
            return

        # Load the best model weights
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        # Run inference on the entire test set
        test_metrics, sp_data, od_data = trainer._run_epoch(test_loader, is_training=False, epoch=-1)
        logger.info(f"Final Test Loss Breakdown: " + ", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()]))
        
        # Unpack predictions and targets for evaluation functions
        sp_preds, sp_targets = sp_data
        od_preds, od_targets = od_data
        
        # Evaluate Signal Processing task
        if sp_preds:
            final_sp_preds = {k: torch.cat([p[k] for p in sp_preds]) for k in sp_preds[0]}
            final_sp_targets = {k: torch.cat([t[k] for t in sp_targets]) for k in sp_targets[0]}
            evaluate_and_visualize_sp([final_sp_preds], [final_sp_targets], logger, result_dir)
        
        # Evaluate Object Detection task
        if od_preds:
            final_od_preds = {k: torch.cat([p[k] for p in od_preds]) for k in od_preds[0]}
            final_od_targets = {k: torch.cat([t[k] for t in od_targets]) for k in od_targets[0]}
            evaluate_and_visualize_od(final_od_preds, final_od_targets, logger, result_dir)
            if sp_preds:
                plot_od_predictions_on_rdm(final_sp_preds, final_od_preds, final_od_targets, SP_CONFIG, logger, result_dir)

        # Save final test metrics to a JSON file
        with open(os.path.join(result_dir, 'test_results.json'), 'w') as f:
            json.dump({'test_metrics': test_metrics}, f, indent=4)

    except Exception:
        logger.error("An unrecoverable error occurred during the experiment:", exc_info=True)
    finally:
        logger.info(f"--- Experiment Finished. Results saved in: {result_dir} ---")
        logging.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a unified radar perception model.")
    parser.add_argument(
        '--model_type', type=str, default='snn_hybrid', 
        choices=['cnn', 'cnn_mlp', 'snn', 'snn_hybrid', 'snn_e2e'],
        help="Type of model to train: cnn, cnn_mlp, snn_hybrid (or snn), snn_e2e."
    )
    main(parser.parse_args())