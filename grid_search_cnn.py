# grid_search_cnn.py (Modified)

"""
Performs a grid search over CNN model hyperparameters.
"""

import os
import torch
import itertools
import pandas as pd
import datetime
import logging
from src.cnn_model import UnifiedRadarCNN
from src.config import SP_CONFIG, RADAR_CONFIG, TRAINING_CONFIG, BASE_DATA_PATH
from src.data_loader import create_data_loaders
from src.trainer import Trainer
from src.utils import setup_logging, save_config

def run_cnn_grid_search():
    param_grid = {
        'learning_rate': [1e-3, 1e-4],
        'hidden_size': [128, 256],
    }

    search_config = TRAINING_CONFIG.copy()
    search_config['num_epochs'] = 3
    search_config['max_frames_per_seq'] = 20
    search_config['dataset_path'] = BASE_DATA_PATH

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    grid_search_dir = os.path.join("results", f"CNN_Grid_Search_{timestamp}")
    os.makedirs(grid_search_dir, exist_ok=True)
    
    main_logger = setup_logging(grid_search_dir)
    main_logger.info("--- Starting CNN Hyperparameter Grid Search ---")

    train_loader, val_loader, _ = create_data_loaders(search_config, SP_CONFIG)

    results = []
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    main_logger.info(f"Testing {len(param_combinations)} parameter combinations...")

    for i, params in enumerate(param_combinations):
        run_name = f"run_{i+1}_" + "_".join([f"{k.replace('_', '')[0:4]}={v}" for k, v in params.items()])
        run_dir = os.path.join(grid_search_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        run_logger = logging.getLogger(run_name)
        # ... (logging setup is the same) ...
        
        main_logger.info(f"\n--- Starting {run_name} ---")

        # Combine base configs with grid search params
        current_params = {**RADAR_CONFIG, **search_config, **SP_CONFIG, **params}
        save_config(current_params, run_dir)

        model = UnifiedRadarCNN(**current_params).to(torch.device(current_params['device']))
        
        trainer = Trainer(model, train_loader, val_loader, current_params, run_dir)
        history = trainer.train(run_logger)

        best_val_loss = min(history['val_total_loss']) if history['val_total_loss'] else float('inf')
        main_logger.info(f"--- Finished {run_name} | Best Val Loss: {best_val_loss:.4f} ---")

        result_summary = params.copy()
        result_summary['best_val_loss'] = best_val_loss
        results.append(result_summary)

    results_df = pd.DataFrame(results).sort_values(by='best_val_loss')
    main_logger.info("\n\n--- Grid Search Complete ---")
    main_logger.info("Results summary:\n" + results_df.to_string())
    results_df.to_csv(os.path.join(grid_search_dir, 'grid_search_summary.csv'), index=False)


if __name__ == '__main__':
    run_cnn_grid_search()
