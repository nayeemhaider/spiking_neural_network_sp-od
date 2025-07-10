"""
Defines traditional (non-spiking) models for comparison.

This file includes two distinct baseline models:

1.  UnifiedRadarCNN: The primary baseline, using a CNN frontend for feature
    extraction and a GRU backend for sequential, time-aware processing. This
    is the direct analog to the recurrent SNNs.

2.  UnifiedRadarCNN_MLP: A simpler, time-agnostic baseline that replaces the
    GRU with a standard MLP. It aggregates features over time (e.g., by
    averaging) before making a prediction, which serves to highlight the
    importance of sequential processing.
"""

import torch
import torch.nn as nn
from .model import SpatialAttention, AnalogObjectDetectionHead, AnalogSPHead # Using the analog heads

# =============================================================================
# MODEL 1: CNN + GRU (Time-Aware Sequential Baseline)
# =============================================================================

class UnifiedRadarCNN(nn.Module):
    """
    An Analog Neural Network (ANN/CNN) version with a GRU for sequence processing.
    """
    def __init__(self, **params):
        super().__init__()
        self.time_steps = params['time_steps']

        self.conv_frontend = nn.Sequential(
            nn.Conv1d(params['num_antennas'], 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU()
        )
        
        conv_out_size = 64 * (params['num_samples'] // 4)
        hidden_size = params['hidden_size']

        # The GRU processes the sequence of features over time.
        self.recurrent_backend = nn.GRU(
            input_size=conv_out_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.od_head = AnalogObjectDetectionHead(hidden_size, params['max_objects'])
        self.sp_head = AnalogSPHead(hidden_size, params['range_bins'], params['doppler_bins'])

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.permute(0, 1, 3, 2).reshape(
            batch_size * self.time_steps, x.shape[3], x.shape[2]
        )
        
        conv_features = self.conv_frontend(x_reshaped)
        conv_features_flat = conv_features.view(batch_size, self.time_steps, -1)
        
        # Pass the sequence of features through the GRU.
        recurrent_features, _ = self.recurrent_backend(conv_features_flat)
        
        # Average the features from all time steps for the final prediction.
        avg_features = recurrent_features.mean(dim=1)
        
        od_outputs = self.od_head(avg_features)
        sp_outputs = self.sp_head(avg_features)
        
        return {'sp_outputs': sp_outputs, 'od_outputs': od_outputs}


# =============================================================================
# MODEL 2: CNN + MLP (Time-Agnostic Baseline)
# =============================================================================

class UnifiedRadarCNN_MLP(nn.Module):
    """
    A simpler, time-agnostic baseline that replaces the GRU with an MLP.
    It aggregates features over time before prediction.
    """
    def __init__(self, **params):
        super().__init__()
        self.time_steps = params['time_steps']

        # CNN frontend is identical to the GRU version
        self.conv_frontend = nn.Sequential(
            nn.Conv1d(params['num_antennas'], 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU()
        )
        
        conv_out_size = 64 * (params['num_samples'] // 4)
        hidden_size = params['hidden_size']

        # The backend is now a simple MLP instead of a GRU.
        self.mlp_backend = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.od_head = AnalogObjectDetectionHead(hidden_size, params['max_objects'])
        self.sp_head = AnalogSPHead(hidden_size, params['range_bins'], params['doppler_bins'])

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.permute(0, 1, 3, 2).reshape(
            batch_size * self.time_steps, x.shape[3], x.shape[2]
        )
        
        # Pass all time steps through the CNN frontend at once.
        conv_features = self.conv_frontend(x_reshaped)
        conv_features_flat = conv_features.view(batch_size, self.time_steps, -1)

        # --- KEY DIFFERENCE ---
        # Aggregate features across the time dimension BEFORE the backend.
        # This collapses the temporal information into a single static vector.
        aggregated_features = conv_features_flat.mean(dim=1)
        
        # Pass this single aggregated vector through the MLP backend.
        mlp_features = self.mlp_backend(aggregated_features)
        
        # Use the final features for the output heads.
        od_outputs = self.od_head(mlp_features)
        sp_outputs = self.sp_head(mlp_features)
        
        return {'sp_outputs': sp_outputs, 'od_outputs': od_outputs}
