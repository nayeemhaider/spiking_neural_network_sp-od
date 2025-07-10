# src/utils.py

"""
Utility functions for the radar perception project.

This module provides helpers for:
- Logging and configuration management.
- Custom loss function computations.
- Plotting training history and evaluation results.
- Detailed evaluation of Signal Processing (SP) and Object Detection (OD) tasks.
"""

import os
import csv
import numpy as np
import json
import logging
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from .config import LABEL_MAP

# =============================================================================
# SETUP, LOGGING & SAVING
# =============================================================================

def setup_logging(result_dir):
    """Initializes logging to both a file and the console."""
    log_filename = os.path.join(result_dir, 'experiment_log.log')
    # Clear existing handlers to avoid duplicate logs in interactive environments
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def save_config(config_dict, result_dir):
    """Saves the configuration dictionary as a JSON file for reproducibility."""
    def convert(o):
        if isinstance(o, torch.device):
            return str(o)
        # Handle other non-serializable types if they appear
        try:
            return o.__dict__
        except AttributeError:
            return f"Object of type {o.__class__.__name__} is not JSON serializable"
        
    with open(os.path.join(result_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4, default=convert, skipkeys=True)

def setup_batch_logger(result_dir):
    """Sets up a CSV logger to record loss components for every training batch."""
    log_file_path = os.path.join(result_dir, 'batch_losses.csv')
    log_file = open(log_file_path, 'w', newline='')
    fieldnames = [
        'epoch', 'batch_idx', 'total_loss', 'rdm_loss', 'cfar_loss', 'exist_loss', 
        'range_loss', 'angle_loss', 'rcs_loss', 'velocity_loss'
    ]
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()
    return log_file, writer

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def compute_sp_loss(predictions, targets):
    """Computes the loss for the Signal Processing (SP) task."""
    rdm_loss = F.mse_loss(predictions['rdm'], targets['rdm'])
    cfar_mask = targets['cfar_mask']
    pos_weight = torch.tensor(1.0, device=cfar_mask.device)
    num_pos = torch.sum(cfar_mask == 1)
    if num_pos > 0:
        num_neg = torch.sum(cfar_mask == 0)
        pos_weight = num_neg / num_pos
    cfar_loss = F.binary_cross_entropy_with_logits(predictions['cfar_logits'], cfar_mask, pos_weight=pos_weight)
    return rdm_loss, cfar_loss

def compute_od_loss(predictions, targets):
    """Computes the loss for the Object Detection (OD) task."""
    exist_loss = F.binary_cross_entropy(predictions['existence'], targets['existence'])
    range_loss, angle_loss, velocity_loss, rcs_loss = [torch.tensor(0.0, device=exist_loss.device)] * 4
    
    mask = targets['existence'] > 0.5
    if mask.any():
        if 'range' in predictions:
            range_loss = F.mse_loss(predictions['range'][mask], targets['range'][mask])
        if 'angle' in predictions:
            angle_loss = F.mse_loss(predictions['angle'][mask], targets['angle'][mask])
        if 'velocity' in predictions:
            velocity_loss = F.mse_loss(predictions['velocity'][mask], targets['velocity'][mask])
        if 'rcs' in predictions:
            rcs_loss = F.mse_loss(predictions['rcs'][mask], targets['rcs'][mask])

    return exist_loss, range_loss, angle_loss, rcs_loss, velocity_loss

# =============================================================================
# EVALUATION & VISUALIZATION
# =============================================================================

def calculate_detection_snr(rdm_abs, mask):
    """
    Estimates the average Signal-to-Noise Ratio (SNR) for detected points.
    """
    detection_mask = mask > 0.5
    signal_pixels = rdm_abs[detection_mask]
    
    if signal_pixels.numel() == 0:
        return 0.0

    noise_floor = rdm_abs[~detection_mask].mean() + 1e-9
    avg_signal = signal_pixels.mean()
    snr_linear = avg_signal / noise_floor
    if snr_linear <= 0: return 0.0
    
    snr_db = 10 * torch.log10(snr_linear)
    return snr_db.item()

def evaluate_and_visualize_sp(predictions, targets, logger, result_dir):
    """
    Evaluates signal processing results, including Precision, Recall, F1, and SNR.
    This version is robust and does not require 'rdm_abs' in the targets dictionary.
    """
    logger.info("--- SP Evaluation (CFAR Mask Performance) ---")
    
    pred_cfar_logits = predictions[0]['cfar_logits']
    target_cfar_masks = targets[0]['cfar_mask']
    
    # Estimate the absolute RDM from the log-normalized ground truth RDM
    # This reverses the log1p operation from the data loader.
    target_rdm_abs_est = torch.expm1(targets[0]['rdm'])
    
    all_true_flat, all_pred_flat = [], []
    all_snr_gt, all_snr_pred = [], []

    for i in range(pred_cfar_logits.shape[0]):
        true_mask_flat = (target_cfar_masks[i] > 0.5).flatten().cpu().numpy()
        
        pred_probas = torch.sigmoid(pred_cfar_logits[i])
        pred_mask_flat = (pred_probas.flatten() > 0.5).cpu().numpy()
        
        all_true_flat.extend(true_mask_flat)
        all_pred_flat.extend(pred_mask_flat)
        
        rdm_sample_for_snr = target_rdm_abs_est[i].cpu()
        
        snr_gt = calculate_detection_snr(rdm_sample_for_snr, target_cfar_masks[i].cpu())
        snr_pred = calculate_detection_snr(rdm_sample_for_snr, (pred_probas > 0.5).cpu())
        
        if snr_gt > 0: all_snr_gt.append(snr_gt)
        if snr_pred > 0: all_snr_pred.append(snr_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_flat, all_pred_flat, average='binary', zero_division=0
    )
    
    avg_snr_gt = np.mean(all_snr_gt) if all_snr_gt else 0.0
    avg_snr_pred = np.mean(all_snr_pred) if all_snr_pred else 0.0

    logger.info("="*50)
    logger.info("CFAR Mask Performance Metrics:")
    logger.info(f"  - Precision:            {precision:.4f}")
    logger.info(f"  - Recall (Sensitivity): {recall:.4f}")
    logger.info(f"  - F1-Score:             {f1:.4f}")
    logger.info("-" * 50)
    logger.info("Signal-to-Noise Ratio (SNR) Analysis:")
    logger.info(f"  - Avg. SNR of GT Detections:  {avg_snr_gt:.2f} dB")
    logger.info(f"  - Avg. SNR of Pred Detections:{avg_snr_pred:.2f} dB")
    logger.info("="*50)

    # Visualization
    idx = 0
    pred_rdm_vis = predictions[0]['rdm'][idx].cpu().numpy()
    target_rdm_vis = targets[0]['rdm'][idx].cpu().numpy()
    pred_cfar_vis = (torch.sigmoid(pred_cfar_logits[idx]) > 0.5).int().cpu().numpy()
    target_cfar_vis = target_cfar_masks[idx].cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes[0, 0].imshow(target_rdm_vis, aspect='auto', origin='lower', cmap='viridis'); axes[0, 0].set_title('Target RDM (Normalized)')
    axes[0, 1].imshow(pred_rdm_vis, aspect='auto', origin='lower', cmap='viridis'); axes[0, 1].set_title('Predicted RDM')
    axes[1, 0].imshow(target_cfar_vis, aspect='auto', origin='lower', cmap='gray'); axes[1, 0].set_title('Target CFAR Mask')
    axes[1, 1].imshow(pred_cfar_vis, aspect='auto', origin='lower', cmap='gray'); axes[1, 1].set_title('Predicted CFAR Mask')
    for ax in axes.flat: ax.set_xlabel('Doppler Bins'); ax.set_ylabel('Range Bins')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(result_dir, 'sp_evaluation.png'), dpi=300)
    plt.close()

def evaluate_and_visualize_od(predictions, targets, logger, result_dir):
    """Evaluates and visualizes object detection results using a matching strategy."""
    logger.info("--- OD Evaluation ---")
    y_true, y_pred = [], []
    
    if 'existence' not in predictions or 'existence' not in targets:
        logger.warning("Existence key not found in predictions/targets. Skipping OD evaluation.")
        return

    for i in range(len(predictions['existence'])):
        pred_exists = predictions['existence'][i] > 0.5
        targ_exists = targets['existence'][i] > 0.5
        
        gt_indices = torch.where(targ_exists)[0]
        pred_indices = torch.where(pred_exists)[0]
        
        matched_gt = set()
        for pred_idx in pred_indices:
            best_match_gt_idx, min_dist = -1, float('inf')
            pred_pos = torch.tensor([predictions['range'][i][pred_idx], predictions['angle'][i][pred_idx]])
            
            for gt_idx in gt_indices:
                if gt_idx in matched_gt: continue
                targ_pos = torch.tensor([targets['range'][i][gt_idx], targets['angle'][i][gt_idx]])
                dist = torch.norm(pred_pos - targ_pos)
                if dist < min_dist:
                    min_dist, best_match_gt_idx = dist, gt_idx
            
            true_class = int(targets['class'][i][best_match_gt_idx].item()) if best_match_gt_idx != -1 else -1
            if best_match_gt_idx != -1 and min_dist < 0.15:
                y_true.append(true_class)
                y_pred.append(true_class)
                matched_gt.add(best_match_gt_idx)
            else:
                y_true.append(-1)
                y_pred.append(2)

        for gt_idx in set(gt_indices.numpy()) - matched_gt:
            y_true.append(int(targets['class'][i][gt_idx].item()))
            y_pred.append(-1)

    if y_true:
        labels = sorted(list(set(y_true + y_pred)))
        class_names = [LABEL_MAP.get(c, 'FP' if c != -1 else 'FN') for c in labels]
        report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0)
        logger.info("OD Classification Report:\n" + report)
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Object Detection Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
        plt.savefig(os.path.join(result_dir, 'od_confusion_matrix.png'), bbox_inches='tight')
        plt.close()

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_training_curves(history, result_dir):
    """Plots and saves all training and validation loss components over epochs."""
    if not history or 'train_total_loss' not in history or not history['train_total_loss']:
        logging.warning("History is empty or incomplete. Skipping plotting training curves.")
        return
        
    epochs = range(1, len(history['train_total_loss']) + 1)
    
    # Dynamically find all loss keys from history
    loss_keys = sorted([k.replace('train_', '') for k in history if k.startswith('train_')])
    
    num_plots = len(loss_keys)
    if num_plots == 0: return
    
    ncols = 4
    nrows = (num_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle('Training & Validation Loss Curves', fontsize=16)
    axes = axes.flatten()
    
    for i, key in enumerate(loss_keys):
        ax = axes[i]
        ax.plot(epochs, history[f'train_{key}'], 'o-', label=f'Train {key}')
        ax.plot(epochs, history[f'val_{key}'], 's-', label=f'Val {key}')
        ax.set_title(key.replace("_", " ").title())
        ax.set_xlabel('Epochs'); ax.set_ylabel('Loss')
        ax.legend(); ax.grid(True)
    
    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(result_dir, 'training_curves.png'))
    plt.close()

def plot_batch_losses(result_dir):
    """Reads the batch_losses.csv file and plots the training loss per batch."""
    batch_log_path = os.path.join(result_dir, 'batch_losses.csv')
    if not os.path.exists(batch_log_path): return
    try:
        df = pd.read_csv(batch_log_path)
        if df.empty: return
        df['global_step'] = df.index
        plt.figure(figsize=(15, 8))
        plt.plot(df['global_step'], df['total_loss'], alpha=0.7, label='Total Loss per Batch')
        plt.plot(df['global_step'], df['total_loss'].rolling(window=50).mean(), color='red', label='Smoothed Trend')
        plt.title('Training Loss per Batch'); plt.xlabel('Training Step'); plt.ylabel('Total Loss (Log Scale)')
        plt.legend(); plt.grid(True); plt.yscale('log')
        plt.savefig(os.path.join(result_dir, 'batch_training_loss_curve.png'))
        plt.close()
    except Exception as e:
        logging.warning(f"Could not plot batch losses: {e}")

def plot_od_predictions_on_rdm(sp_preds, od_preds, od_targets, sp_config, logger, result_dir):
    """Visualizes predicted and ground truth object detections on the predicted RDM."""
    idx = 0
    if not sp_preds or 'rdm' not in sp_preds or sp_preds['rdm'].shape[0] == 0:
        logger.warning("Cannot plot OD predictions, SP prediction data is empty or missing 'rdm' key.")
        return
        
    predicted_rdm = sp_preds['rdm'][idx].cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(predicted_rdm, aspect='auto', origin='lower', cmap='viridis',
                   extent=[0, sp_config['doppler_bins'], 0, sp_config['range_bins']])
    fig.colorbar(im, ax=ax)
    ax.set_title('Predicted Detections on Predicted RDM (Test Sample)'); ax.set_xlabel('Doppler Bins'); ax.set_ylabel('Range Bins')

    # Draw GROUND TRUTH detections (Green Circles)
    for i in range(od_targets['existence'].shape[1]):
        if od_targets['existence'][idx, i] > 0.5:
            gt_range = od_targets['range'][idx, i] * sp_config['range_bins']
            gt_doppler = (od_targets['velocity'][idx, i] * (sp_config['doppler_bins'] / 2)) + (sp_config['doppler_bins'] / 2)
            circle = patches.Circle((gt_doppler, gt_range), radius=3, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(circle)

    # Draw PREDICTED detections (Red Squares)
    for i in range(od_preds['existence'].shape[1]):
        if od_preds['existence'][idx, i] > 0.5:
            pred_range = od_preds['range'][idx, i] * sp_config['range_bins']
            pred_doppler = (od_preds['velocity'][idx, i] * (sp_config['doppler_bins'] / 2)) + (sp_config['doppler_bins'] / 2)
            rect = patches.Rectangle((pred_doppler-2, pred_range-2), 4, 4, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    plt.tight_layout()
    save_path = os.path.join(result_dir, 'od_predictions_visualization.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Object detection visualization saved to: {save_path}")