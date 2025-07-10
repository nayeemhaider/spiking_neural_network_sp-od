"""
Defines the Trainer class for handling model training and validation loops.

The Trainer encapsulates the optimizer, learning rate scheduler, loss computation,
epoch execution, model saving, and early stopping logic.
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm
from .utils import compute_sp_loss, compute_od_loss, setup_batch_logger

class Trainer:
    """Manages the training and validation process for a unified radar model."""
    def __init__(self, model, train_loader, val_loader, config, result_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.result_dir = result_dir
        self.optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.0001))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.sp_weight = config.get('sp_loss_weight', 1.0)
        self.od_weight = config.get('od_loss_weight', 1.0)

    def _compute_unified_loss(self, predictions, sp_targets, od_targets):
        """Computes the weighted, unified loss for both SP and OD tasks."""
        sp_preds = predictions['sp_outputs']
        od_preds = predictions['od_outputs']
        
        rdm_loss, cfar_loss = compute_sp_loss(sp_preds, sp_targets)
        
        # Updated unpacking
        exist_loss, range_loss, angle_loss, rcs_loss, velocity_loss = compute_od_loss(od_preds, od_targets)
        
        sp_total_loss = rdm_loss + cfar_loss
        # Updated total OD loss calculation
        od_total_loss = exist_loss + range_loss + angle_loss + rcs_loss + velocity_loss
        
        total_loss = (self.sp_weight * sp_total_loss) + (self.od_weight * od_total_loss)
        
        # Updated dictionary of losses to return
        return {
            'total_loss': total_loss, 'rdm_loss': rdm_loss, 'cfar_loss': cfar_loss,
            'exist_loss': exist_loss, 'range_loss': range_loss, 'angle_loss': angle_loss,
            'rcs_loss': rcs_loss, 'velocity_loss': velocity_loss
        }

    def _run_epoch(self, data_loader, is_training, epoch, batch_logger_writer=None):
        """Runs a single epoch of training or validation."""
        self.model.train(is_training)
        
        # Updated dictionary of epoch losses to track
        epoch_losses = {
            'total_loss': 0.0, 'rdm_loss': 0.0, 'cfar_loss': 0.0, 'exist_loss': 0.0,
            'range_loss': 0.0, 'angle_loss': 0.0, 'rcs_loss': 0.0, 'velocity_loss': 0.0
        }
        
        all_sp_preds, all_sp_targets, all_od_preds, all_od_targets = [], [], [], []
        
        desc = "Training" if is_training else "Validation"
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} {desc}", leave=False)
        
        with torch.set_grad_enabled(is_training):
            for batch_idx, batch_data in enumerate(progress_bar):
                if not batch_data or len(batch_data) < 3: continue
                snn_input, sp_targets, od_targets = batch_data

                snn_input = snn_input.to(self.device)
                sp_targets = {k: v.to(self.device) for k, v in sp_targets.items()}
                od_targets = {k: v.to(self.device) for k, v in od_targets.items()}
                
                if is_training: self.optimizer.zero_grad()
                
                predictions = self.model(snn_input)
                loss_dict = self._compute_unified_loss(predictions, sp_targets, od_targets)
                
                if is_training and batch_logger_writer:
                    log_row = {'epoch': epoch, 'batch_idx': batch_idx}
                    log_row.update({k: v.item() for k, v in loss_dict.items()})
                    batch_logger_writer.writerow(log_row)
                
                if is_training:
                    loss_dict['total_loss'].backward()
                    self.optimizer.step()

                for key in epoch_losses:
                    if key in loss_dict: epoch_losses[key] += loss_dict[key].item()
                
                progress_bar.set_postfix({'Loss': f"{loss_dict['total_loss'].item():.4f}"})

                if not is_training:
                    all_sp_preds.append({k: v.cpu().detach() for k, v in predictions['sp_outputs'].items()})
                    all_sp_targets.append({k: v.cpu().detach() for k, v in sp_targets.items()})
                    all_od_preds.append({k: v.cpu().detach() for k, v in predictions['od_outputs'].items()})
                    all_od_targets.append({k: v.cpu().detach() for k, v in od_targets.items()})
        
        avg_losses = {k: v / len(data_loader) for k, v in epoch_losses.items()}
        return avg_losses, (all_sp_preds, all_sp_targets), (all_od_preds, all_od_targets)

    def train(self, logger):
        """The main training loop."""
        # Updated list of keys to initialize history dictionary
        loss_keys = [
            'total_loss', 'rdm_loss', 'cfar_loss', 'exist_loss', 'range_loss',
            'angle_loss', 'rcs_loss', 'velocity_loss'
        ]
        history = {f'{phase}_{key}': [] for phase in ['train', 'val'] for key in loss_keys}

        batch_log_file, batch_logger_writer = setup_batch_logger(self.result_dir)
        best_val_loss = float('inf')
        
        try:
            for epoch in range(1, self.config.get('num_epochs', 10) + 1):
                train_metrics, _, _ = self._run_epoch(self.train_loader, is_training=True, epoch=epoch, batch_logger_writer=batch_logger_writer)
                val_metrics, _, _ = self._run_epoch(self.val_loader, is_training=False, epoch=epoch)
                
                self.scheduler.step(val_metrics['total_loss'])
                
                logger.info(f"Epoch {epoch} | Train Loss: {train_metrics['total_loss']:.4f} | Val Loss: {val_metrics['total_loss']:.4f}")
                
                for key in train_metrics:
                    if f'train_{key}' in history:
                        history[f'train_{key}'].append(train_metrics[key])
                        history[f'val_{key}'].append(val_metrics[key])

                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    torch.save(self.model.state_dict(), os.path.join(self.result_dir, 'best_model.pth'))
                    logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        finally:
            if batch_log_file: batch_log_file.close()
                
        return history
    
