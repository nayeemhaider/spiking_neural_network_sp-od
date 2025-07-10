"""
Handles data loading, preprocessing, and ground truth generation.

This module includes:
- A PyTorch Dataset class (`UnifiedRadarDataset`) that processes raw radar
  frames and generates corresponding ground truth labels for both Signal
  Processing (SP) and Object Detection (OD) tasks on-the-fly.
- Helper functions to load file paths from the dataset directory and to
  create balanced data loaders for training, validation, and testing.
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def run_2d_cfar(rdm_db, train_cells, guard_cells, threshold_db):
    """
    Generates a 2D Constant False Alarm Rate (CFAR) mask from a Range-Doppler Map.

    Args:
        rdm_db (torch.Tensor): RDM in dB scale.
        train_cells (int): Number of training cells for noise estimation.
        guard_cells (int): Number of guard cells to exclude around the cell under test.
        threshold_db (float): Detection threshold in dB above the estimated noise.

    Returns:
        torch.Tensor: A binary mask where 1 indicates a detection.
    """
    win_size = 2 * (train_cells + guard_cells) + 1
    # Create a kernel for averaging with a hole in the middle for guard cells.
    kernel = torch.ones(win_size, win_size, device=rdm_db.device)
    center_start, center_end = train_cells, win_size - train_cells
    kernel[center_start:center_end, center_start:center_end] = 0
    
    num_noise_cells = win_size**2 - (2 * guard_cells + 1)**2
    if num_noise_cells <= 0:
        return torch.zeros_like(rdm_db)

    # Use 2D convolution to efficiently calculate the average noise level for each cell.
    noise_level = F.conv2d(rdm_db.unsqueeze(0).unsqueeze(0),
                           kernel.unsqueeze(0).unsqueeze(0), padding='same') / num_noise_cells
                           
    return (rdm_db > noise_level.squeeze() + threshold_db).float()


def load_real_radar_data(dataset_path, sequence_names, max_frames_per_seq):
    """
    Loads raw radar data and corresponding text labels from multiple sequences.

    Args:
        dataset_path (str): Root path of the dataset.
        sequence_names (list): List of sequence folders to load.
        max_frames_per_seq (int): Maximum number of frames to load per sequence.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of all loaded radar data frames.
            - list: A list of all corresponding frame labels.
    """
    all_radar_data, all_labels = [], []
    for seq_name in sequence_names:
        print(f"Loading sequence: {seq_name}")
        sequence_path = os.path.join(dataset_path, seq_name)
        radar_path = os.path.join(sequence_path, "radar_raw_frame")
        label_path = os.path.join(sequence_path, "text_labels")
        if not os.path.exists(radar_path) or not os.path.exists(label_path):
            print(f"Warning: Data path not found for {seq_name}, skipping.")
            continue
        
        radar_files = sorted(glob.glob(os.path.join(radar_path, "*.mat")))
        if max_frames_per_seq:
            radar_files = radar_files[:max_frames_per_seq]
        if not radar_files:
            continue

        for radar_file in tqdm(radar_files, desc=f"Frames from {seq_name}"):
            try:
                mat_data = scipy.io.loadmat(radar_file)
                # Handle different key names for ADC data in .mat files.
                adc_data = mat_data.get('adcData', mat_data.get('adc_data'))
                if adc_data is None or adc_data.ndim != 4:
                    continue

                all_radar_data.append(adc_data)
                
                frame_name = os.path.basename(radar_file).replace('.mat', '')
                label_file = os.path.join(label_path, f"{frame_name}.csv")
                frame_labels = []
                if os.path.exists(label_file):
                    df = pd.read_csv(label_file, header=None, names=['uid', 'class', 'px', 'py', 'wid', 'len'], sep='[;,]', engine='python')
                    for _, row in df.iterrows():
                        try:
                            frame_labels.append({
                                'class': int(row['class']), 'px': float(row['px']), 'py': float(row['py']),
                                'wid': float(row['wid']), 'len': float(row['len'])
                            })
                        except (ValueError, TypeError):
                            continue
                all_labels.append(frame_labels)
            except Exception as e:
                print(f"Error loading {os.path.basename(radar_file)}: {e}")

    return np.array(all_radar_data), all_labels


class UnifiedRadarDataset(Dataset):
    """
    PyTorch Dataset for unified radar perception.

    Processes raw complex ADC data on-the-fly to generate:
    1. Normalized magnitude tensor as input for the SNN.
    2. Ground truth for the Signal Processing (SP) task (RDM and CFAR mask).
    3. Ground truth for the Object Detection (OD) task (padded and normalized).
    """
    def __init__(self, radar_data, od_labels, time_steps, sp_config, train_config):
        self.radar_data = torch.tensor(radar_data, dtype=torch.complex64)
        self.od_labels = od_labels
        self.time_steps = time_steps
        self.sp_config = sp_config
        self.train_config = train_config
    
    def __len__(self):
        return len(self.radar_data)

    def __getitem__(self, idx):
        # --- 1. SNN Input Preparation ---
        radar_frame = self.radar_data[idx]
        samples, chirps, receivers, transmitters = radar_frame.shape
        virtual_antennas = receivers * transmitters
        
        # Reshape to (chirps, samples, virtual_antennas)
        radar_reshaped = radar_frame.permute(1, 0, 2, 3).reshape(chirps, samples, virtual_antennas)
        
        # Sample or pad chirps to match the model's required time_steps.
        if chirps >= self.time_steps:
            indices = torch.linspace(0, chirps - 1, self.time_steps).long()
            snn_input_complex = radar_reshaped[indices]
        else:
            snn_input_complex = torch.zeros(self.time_steps, samples, virtual_antennas, dtype=torch.complex64)
            snn_input_complex[:chirps] = radar_reshaped

        # Use magnitude as input and normalize it.
        snn_input_mag = torch.abs(snn_input_complex)
        snn_input = (snn_input_mag - snn_input_mag.mean()) / (snn_input_mag.std() + 1e-8)

        # --- 2. Signal Processing (SP) Ground Truth Generation ---
        # Use the first virtual antenna's data to generate the RDM.
        sp_data = snn_input_complex[:, :, 0]
        range_fft = torch.fft.fft(sp_data, n=self.sp_config['range_bins'], dim=1)
        doppler_fft = torch.fft.fft(range_fft, n=self.sp_config['doppler_bins'], dim=0)
        rdm = torch.fft.fftshift(doppler_fft, dim=0)
        
        target_rdm_abs = torch.abs(rdm).T  # Transpose to (range, doppler)
        target_rdm_log = torch.log1p(target_rdm_abs)
        target_rdm_norm = (target_rdm_log - target_rdm_log.min()) / (target_rdm_log.max() - target_rdm_log.min() + 1e-8)
        
        # Generate the CFAR mask from the RDM.
        rdm_db = 20 * torch.log10(target_rdm_abs + 1e-8)
        cfar_params = {
            'train_cells': self.sp_config['cfar_train_cells'],
            'guard_cells': self.sp_config['cfar_guard_cells'],
            'threshold_db': self.sp_config['cfar_threshold_dB']
        }
        target_cfar_mask = run_2d_cfar(rdm_db, **cfar_params)
        
        sp_targets = {'rdm': target_rdm_norm, 'cfar_mask': target_cfar_mask}

        # --- 3. Object Detection (OD) Ground Truth Generation ---
        frame_od_labels = []
        max_objects = self.train_config['max_objects']
        
        # Define normalization constants for object properties.
        max_range, max_angle_degrees, max_rcs = 24.0, 90.0, 50.0
        max_width, max_length = 5.0, 10.0

        for obj in self.od_labels[idx][:max_objects]:
            px, py = obj['px'], obj['py']
            obj_range = np.sqrt(px**2 + py**2)
            obj_angle = np.arctan2(px, py) * 180 / np.pi
            obj_wid = obj.get('wid', 1.0)
            obj_len = obj.get('len', 1.0)
            
            # Estimate velocity by finding the peak in the RDM at the object's range.
            range_bin_idx = min(int((obj_range / max_range) * self.sp_config['range_bins']), self.sp_config['range_bins'] - 1)
            if range_bin_idx < target_rdm_abs.shape[0]:
                doppler_profile = target_rdm_abs[range_bin_idx, :]
                doppler_bin_idx = torch.argmax(doppler_profile).item()
                # Normalize velocity to [-1, 1] for the tanh activation.
                normalized_velocity = (doppler_bin_idx - (self.sp_config['doppler_bins'] / 2)) / (self.sp_config['doppler_bins'] / 2)
            else:
                normalized_velocity = 0.0

            rcs_map = {0: 1.0, 2: 10.0, 3: 5.0, 5: 25.0, 7: 30.0, 80: 2.0}
            estimated_rcs = rcs_map.get(obj['class'], 5.0) + obj_wid * obj_len
            
            frame_od_labels.append({
                'existence': 1.0, 'class': obj['class'],
                'range': min(obj_range / max_range, 1.0),
                'angle': np.clip(obj_angle / max_angle_degrees, -1.0, 1.0),
                'velocity': normalized_velocity,
                'width': min(obj_wid / max_width, 1.0),
                'length': min(obj_len / max_length, 1.0),
                'rcs': min(estimated_rcs / max_rcs, 1.0)
            })

        # Pad with non-existent objects up to max_objects.
        while len(frame_od_labels) < max_objects:
            frame_od_labels.append({
                'existence': 0.0, 'class': -1, 'range': 0.0, 'angle': 0.0, 
                'velocity': 0.0, 'width': 0.0, 'length': 0.0, 'rcs': 0.0
            })

        # Convert the list of dicts to a single dict of tensors.
        od_targets = {key: torch.tensor([obj[key] for obj in frame_od_labels], dtype=torch.float32) for key in frame_od_labels[0]}

        return snn_input, sp_targets, od_targets


def create_data_loaders(config, sp_config):
    """
    Main function to load data and create train/val/test dataloaders.

    Args:
        config (dict): The main training configuration dictionary.
        sp_config (dict): The signal processing configuration dictionary.

    Returns:
        tuple: A tuple containing the train, validation, and test DataLoaders.
    """
    print("Loading raw radar data and labels...")
    radar_data, labels = load_real_radar_data(
        config['dataset_path'], config['sequence_names'], config['max_frames_per_seq']
    )
    if len(radar_data) == 0:
        raise FileNotFoundError("No data loaded. Check dataset paths and sequence names in config.py")
    
    indices = np.arange(len(labels))
    
    # Split data indices into train, validation, and test sets (70/20/10 split).
    train_val_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2 / 0.9, random_state=42)

    train_dataset = UnifiedRadarDataset(radar_data[train_indices], [labels[i] for i in train_indices], config['time_steps'], sp_config, config)
    val_dataset = UnifiedRadarDataset(radar_data[val_indices], [labels[i] for i in val_indices], config['time_steps'], sp_config, config)
    test_dataset = UnifiedRadarDataset(radar_data[test_indices], [labels[i] for i in test_indices], config['time_steps'], sp_config, config)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    
    print(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    return train_loader, val_loader, test_loader

