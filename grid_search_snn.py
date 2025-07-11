"""
Performs a grid search over SNN model and neuron hyperparameters.
Supports both the hybrid and end-to-end SNN models.
"""

import os
import torch
import itertools
import pandas as pd
import datetime
import logging
import argparse
from src.model import UnifiedRadarSNN_Hybrid, UnifiedRadarSNN_End2End
from src.config import SP_CONFIG, RADAR_CONFIG, TRAINING_CONFIG, BASE_DATA_PATH
from src.data_loader import create_data_loaders
from src.trainer import Trainer
from src.utils import setup_logging, save_config

def run_snn_grid_search(args):
    """
    Defines the hyperparameter grid and runs the search process for the selected SNN model.
    """
    if args.model_type == 'snn_hybrid':
        model_class = UnifiedRadarSNN_Hybrid
        name_prefix = "SNN_Hybrid_GRU" # Use a more descriptive name
    elif args.model_type == 'snn_e2e':
        model_class = UnifiedRadarSNN_End2End
        name_prefix = "SNN_End2End"
    else:
        raise ValueError("Invalid model type for SNN grid search.")

    param_grid = {
        'neuron_threshold': [0.8, 1.0, 1.2],     # Adjusted for more reasonable spiking
        'neuron_alpha': [0.5, 0.3, 0.7],         # Membrane potential decay (leak)
        'neuron_beta': [0.5, 0.9, 0.7]           # Higher beta is common for integration             
    }

    # Configure search settings
    search_config = TRAINING_CONFIG.copy()
    search_config['num_epochs'] = 3
    search_config['max_frames_per_seq'] = 20
    search_config['dataset_path'] = BASE_DATA_PATH

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    grid_search_dir = os.path.join("results", f"{name_prefix}_Grid_Search_{timestamp}")
    os.makedirs(grid_search_dir, exist_ok=True)
    
    main_logger = setup_logging(grid_search_dir)
    main_logger.info(f"--- Starting Grid Search for {name_prefix} ---")

    # Load data once for all runs
    try:
        train_loader, val_loader, _ = create_data_loaders(search_config, SP_CONFIG)
    except Exception as e:
        main_logger.error(f"Failed to load data. Error: {e}")
        return

    results = []
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    main_logger.info(f"Testing {len(param_combinations)} parameter combinations...")

    for i, params in enumerate(param_combinations):
        # Create a readable name for the run directory
        run_name = f"run_{i+1:02d}_" + "_".join([f"{k.replace('_', '')[:4]}={v}" for k, v in params.items()])
        run_dir = os.path.join(grid_search_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Set up a dedicated logger for this specific run
        run_logger = logging.getLogger(run_name)
        handler = logging.FileHandler(os.path.join(run_dir, 'run.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not run_logger.hasHandlers():
            run_logger.addHandler(handler)
        
        main_logger.info(f"\n--- Starting {run_name} ---")
        run_logger.info(f"Parameters: {params}")

        # Combine all configurations into a single dictionary
        current_params = {**RADAR_CONFIG, **search_config, **SP_CONFIG, **params}
        
        # This standardizes the key name before passing it to the constructor.
        if 'numAdcSamples' in current_params:
            current_params['num_samples'] = current_params['numAdcSamples']
        
        save_config(current_params, run_dir)
        
        try:
            # Initialize model with the complete set of parameters
            model = model_class(**current_params).to(torch.device(current_params['device']))
            run_logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
            
            # Train the model
            trainer = Trainer(model, train_loader, val_loader, current_params, run_dir)
            history = trainer.train(run_logger)

            # Log results
            best_val_loss = min(history['val_total_loss']) if history['val_total_loss'] else float('inf')
            main_logger.info(f"--- Finished {run_name} | Best Val Loss: {best_val_loss:.4f} ---")

            result_summary = params.copy()
            result_summary['best_val_loss'] = best_val_loss
            results.append(result_summary)

        except Exception as e:
            main_logger.error(f"!!! Run {run_name} FAILED with error: {e}", exc_info=True)
            result_summary = params.copy()
            result_summary['best_val_loss'] = float('inf') # Mark failed runs
            results.append(result_summary)
        
        finally:
            # Clean up the handler for the next run to avoid duplicate logging
            run_logger.removeHandler(handler)
            handler.close()

    # --- Final Summary ---
    if not results:
        main_logger.warning("No runs were completed. Check for errors.")
        return

    results_df = pd.DataFrame(results).sort_values(by='best_val_loss')
    main_logger.info("\n\n--- Grid Search Complete ---")
    main_logger.info("Results summary (sorted by best validation loss):\n" + results_df.to_string())
    
    # Save the summary to a CSV file
    results_df.to_csv(os.path.join(grid_search_dir, 'grid_search_summary.csv'), index=False)
    main_logger.info(f"\nSummary saved to: {os.path.join(grid_search_dir, 'grid_search_summary.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a grid search for SNN models.")
    parser.add_argument('--model_type', type=str, default='snn_hybrid',
                        choices=['snn_hybrid', 'snn_e2e'],
                        help="Which SNN architecture to perform the grid search on.")
    run_snn_grid_search(parser.parse_args())