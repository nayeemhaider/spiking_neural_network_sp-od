"""
Central configuration file for the Unified Radar Perception project.

This file contains all static configurations, including radar specifications,
dataset paths, and hyperparameters for model training and signal processing.
Paths are constructed to be OS-agnostic where possible.
"""

import torch
import os

# --- PATH CONFIGURATION ---
# Get the absolute path to the project's root directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define the base path to the dataset.
# The 'r' prefix treats backslashes as literal characters, which is safe for Windows paths.
BASE_DATA_PATH = r"D:\Automotive"

# Define the base directory where all experiment results will be saved.
RESULT_BASE_DIR = os.path.join(PROJECT_ROOT, 'results')

# --- Path Verification (for immediate user feedback) ---
# Printing on import is generally a side-effect to be avoided in libraries, but it is useful here for immediate feedback when running experiments.
print("-" * 50)
print(f"Project Root Directory: {PROJECT_ROOT}")
print(f"Expecting Dataset at:   {BASE_DATA_PATH}")
print(f"Saving Results to:      {RESULT_BASE_DIR}")
if not os.path.exists(BASE_DATA_PATH):
    print("\n!!!!!! CRITICAL WARNING !!!!!!")
    print(f"The specified dataset path does not exist: {BASE_DATA_PATH}")
    print("Please ensure the drive is connected and the path is correct.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
print("-" * 50)


# --- RADAR & DATASET SPECIFICATIONS ---
RADAR_CONFIG = {
    "startFreqConst_GHz": 77.0,
    "bandwidth_GHz": 0.67,
    "chirpDuration_usec": 60.0,
    "freqSlopeConst_MHz_usec": 21.0,
    "numAdcSamples": 128,               # Samples per chirp, determines range resolution
    "digOutSampleRate": 4000.0,
    "numLoops": 255,                    # Chirps per frame, determines Doppler resolution
    "framePeriodicity_msec": 33.33333,
    "num_antennas": 8, # Number of virtual antennas (2 TX * 4 RX)
}

# Mapping from class ID in the dataset to a human-readable name.
LABEL_MAP = {
    0: 'person', 2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck', 80: 'cyclist'
}

# --- MODEL & TRAINING CONFIGURATION ---
TRAINING_CONFIG = {
    # List of sequences to be used for training, validation, and testing.
    'sequence_names': [
        "2019_04_09_bms1000_Cyclist", 
        "2019_04_09_cms1000_Car", 
        "2019_04_09_pms1000_Person",
        "2019_05_29_pbms007_Person_Cyclist", 
        "2019_05_09_pcms002_Person_Car",
        "2019_05_09_pbms004_Cyclist_Car", 
        "2019_05_29_mlms006_Car_Person_Cyclist"
        # "2019_04_30_mlms001_Person_Cyclist_Car" 
        # "2019_05_29_pcms005_Truck"
    ],

    'max_frames_per_seq': 200,      # Max frames to load per sequence
    'batch_size': 4,
    'learning_rate': 0.0001,
    'num_epochs': 20,
    'time_steps': 255,              # Number of chirps to process per frame (should match numLoops)
    'hidden_size': 256,             # Size of the recurrent layer's hidden state
    'max_objects': 100,             # Maximum number of objects to detect per frame
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,               # Set to > 0 to use multi-process data loading
    'early_stopping_patience': 10,  # Num epochs to wait for improvement before stopping
    'sp_loss_weight': 1.0,          # Weight for the signal processing loss component
    'od_loss_weight': 1.0,          # Weight for the object detection loss component

    'neuron_threshold': 1.0,    # threshold
    'neuron_alpha': 0.3,        # current damping constant
    'neuron_beta': 0.9,         # potential damping constant
    'sg_alpha': 2.0
}

# --- SIGNAL PROCESSING (SP) TARGET GENERATION CONFIG ---
SP_CONFIG = {
    "time_steps": TRAINING_CONFIG['time_steps'],
    "range_bins": RADAR_CONFIG['numAdcSamples'],
    "doppler_bins": TRAINING_CONFIG['time_steps'],
    # Parameters for the 2D CFAR algorithm used to generate ground truth
    "cfar_train_cells": 8,          # Training cells in one dimension for noise estimation
    "cfar_guard_cells": 2,          # Guard cells in one dimension to prevent target leakage
    "cfar_threshold_dB": 15.0,      # Detection threshold in dB above the noise floor
}