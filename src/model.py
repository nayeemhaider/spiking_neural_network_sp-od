"""
Defines Spiking Neural Network (SNN) architectures for radar perception.

This file provides two distinct SNN models for comparative analysis, both of which
are fully parameterizable to allow for systematic study of neuron dynamics via
a grid search script.

1.  UnifiedRadarSNN_Hybrid: A hybrid model with a spiking core and analog decoders (MLP).
2.  UnifiedRadarSNN_End2End: A conceptually pure, end-to-end spiking model.

Spiking Neural Network consist of convolutional frontend which is responsible for feature extraction and
fully connected backend which is responsible to learn the pattern from the features and generate spike trains which will be transmitted
to the output layer for predictions (Signal Processing and Object Detection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# SHARED SNN COMPONENTS
# =============================================================================

class SurrogateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, alpha=5.0):
        ctx.save_for_backward(input)
        ctx.threshold, ctx.alpha = threshold, alpha
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output * ctx.alpha * torch.sigmoid(ctx.alpha * (input - ctx.threshold)) * \
               (1 - torch.sigmoid(ctx.alpha * (input - ctx.threshold)))
        return grad, None, None

class ResonateFireLayer(nn.Module):
    def __init__(self, input_size, hidden_size, **neuron_params):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.threshold = neuron_params.get('neuron_threshold', 1.0)
        self.alpha = neuron_params.get('neuron_alpha', 0.5)  # Damping/decay for current
        self.beta = neuron_params.get('neuron_beta', 0.9)   # Damping/decay for potential
        self.sg_alpha = neuron_params.get('sg_alpha', 5.0)
        self.register_buffer('potential', None)
        self.register_buffer('current', None)

    def reset_state(self, batch_size=1, device='cpu'):
        self.potential = torch.zeros(batch_size, self.linear.out_features, device=device)
        self.current = torch.zeros(batch_size, self.linear.out_features, device=device)

    def forward(self, x):
        self.current = self.alpha * self.current.detach() + self.linear(x)
        self.potential = self.beta * self.potential.detach() + self.current
        spikes = SurrogateGradient.apply(self.potential, self.threshold, self.sg_alpha)
        self.potential = self.potential * (1 - spikes)
        return spikes

class SpikingConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **neuron_params):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.threshold = neuron_params.get('neuron_threshold', 1.0)
        self.alpha = neuron_params.get('neuron_alpha', 0.5)
        self.beta = neuron_params.get('neuron_beta', 0.9)
        self.sg_alpha = neuron_params.get('sg_alpha', 5.0)
        self.register_buffer('potential', None)
        self.register_buffer('current', None)

    def reset_state(self, batch_size=1, length=1, device='cpu'):
        out_shape = (batch_size, self.conv.out_channels, length)
        self.potential = torch.zeros(out_shape, device=device)
        self.current = torch.zeros(out_shape, device=device)

    def forward(self, x):
        self.current = self.alpha * self.current.detach() + self.conv(x)
        self.potential = self.beta * self.potential.detach() + self.current
        spikes = SurrogateGradient.apply(self.potential, self.threshold, self.sg_alpha)
        self.potential = self.potential * (1 - spikes)
        return spikes

class SpatialAttention(nn.Module):
    """ An analog attention module to weigh temporal features. """
    def __init__(self, hidden_size):
        super().__init__()
        # This attention mechanism is analog (non-spiking) for stability and simplicity.
        # It operates on the time-averaged features before decoding.
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )
    def forward(self, x_avg):
        # x_avg is shape (batch, features)
        scores = self.attention_net(x_avg)
        return x_avg * scores

# =============================================================================
# HYBRID SNN (SNN Core + GRU Decoders)
# =============================================================================

class AnalogObjectDetectionHead(nn.Module):
    def __init__(self, hidden_size, max_objects):
        super().__init__()
        self.existence = nn.Linear(hidden_size, max_objects)
        self.attributes = nn.Linear(hidden_size, max_objects * 4)

    def forward(self, x_avg):
        existence = torch.sigmoid(self.existence(x_avg))
        attrs = self.attributes(x_avg).view(x_avg.shape[0], -1, 4)
        return {'existence': existence, 'range': torch.sigmoid(attrs[:,:,0]), 'angle': torch.tanh(attrs[:,:,1]), 'velocity': torch.tanh(attrs[:,:,2]), 'rcs': torch.sigmoid(attrs[:,:,3])}

class AnalogSPHead(nn.Module):
    def __init__(self, hidden_size, range_bins, doppler_bins):
        super().__init__()
        self.range_bins, self.doppler_bins = range_bins, doppler_bins
        output_dim = range_bins * doppler_bins
        self.rdm_head = nn.Linear(hidden_size, output_dim)
        self.cfar_head = nn.Linear(hidden_size, output_dim)

    def forward(self, x_avg):
        rdm = torch.sigmoid(self.rdm_head(x_avg)).view(-1, self.range_bins, self.doppler_bins)
        cfar_logits = self.cfar_head(x_avg).view(-1, self.range_bins, self.doppler_bins)
        return {'rdm': rdm, 'cfar_logits': cfar_logits}

class UnifiedRadarSNN_Hybrid(nn.Module):
    """
    The HYBRID SNN model, now enhanced with a GRU decoder to better process
    temporal information from the spiking core before final prediction.
    """
    def __init__(self, **params):
        super().__init__()
        self.time_steps = params['time_steps']
        self.num_samples = params['num_samples']
        hidden_size = params['hidden_size']
        num_antennas = params['num_antennas']
        
        neuron_params = {k: v for k, v in params.items() if 'neuron' in k or 'sg' in k}
        
        conv_out_len = self.num_samples // 4
        self.sconv1 = SpikingConv1d(num_antennas, 32, 7, 2, 3, **neuron_params)
        self.sconv2 = SpikingConv1d(32, 64, 5, 2, 2, **neuron_params)
        self.input_layer = nn.Linear(64 * conv_out_len, hidden_size)
        self.rf_layer1 = ResonateFireLayer(hidden_size, hidden_size, **neuron_params)

        # This GRU will process the sequence of hidden spike vectors.
        self.gru_decoder = nn.GRU(
            input_size=hidden_size,      # Input features are from the RF layer
            hidden_size=hidden_size,     # Output features have the same dimension
            num_layers=1,                # A single GRU layer is often sufficient
            batch_first=True,            # Crucial for (B, T, F) tensor shape
            bidirectional=False          # We process time in one direction
        )
        
        self.attention = SpatialAttention(hidden_size)
        
        self.od_head = AnalogObjectDetectionHead(hidden_size, params['max_objects'])
        self.sp_head = AnalogSPHead(hidden_size, params['range_bins'], params['doppler_bins'])

    def _reset_states(self, batch_size, device):
        conv_out_len = self.num_samples // 4
        self.sconv1.reset_state(batch_size, self.num_samples // 2, device)
        self.sconv2.reset_state(batch_size, conv_out_len, device)
        self.rf_layer1.reset_state(batch_size, device)

    def forward(self, x):
        batch_size, device = x.shape[0], x.device
        self._reset_states(batch_size, device)
        
        hidden_spikes_over_time = []
        for t in range(self.time_steps):
            x_t = x.permute(0, 1, 3, 2)[:, t, :, :]
            s_c1 = self.sconv1(x_t)
            s_c2 = self.sconv2(s_c1)
            s_c2_flat = s_c2.view(batch_size, -1)
            s_in = F.relu(self.input_layer(s_c2_flat))
            s_r1 = self.rf_layer1(s_in)
            hidden_spikes_over_time.append(s_r1)
        
        # The GRU processes the entire sequence of hidden spike vectors.
        # Its final hidden state serves as the definitive summary of the sequence.
        # gru_output shape: (batch, time_steps, hidden_size)
        # final_hidden_state shape: (num_layers, batch, hidden_size)
        hidden_features_seq = torch.stack(hidden_spikes_over_time, dim=1)
        _, final_hidden_state = self.gru_decoder(hidden_features_seq)

        # We take the hidden state from the last layer and remove the first dimension.
        # This is our new summary vector, rich with temporal information.
        summary_vector = final_hidden_state.squeeze(0)
        
        od_outputs = self.od_head(summary_vector)
        sp_outputs = self.sp_head(summary_vector)
        
        return {'sp_outputs': sp_outputs, 'od_outputs': od_outputs}

# =============================================================================
# END-TO-END SNN (SNN Core + SNN Decoders)
# =============================================================================

class SpikingObjectDetectionHead(nn.Module):
    def __init__(self, hidden_size, max_objects, **neuron_params):
        super().__init__()
        self.num_attrs = 4
        self.decoder = ResonateFireLayer(hidden_size, max_objects * self.num_attrs, **neuron_params)

    def forward(self, hidden_features_seq):
        batch_size, time_steps, _ = hidden_features_seq.shape
        self.decoder.reset_state(batch_size, hidden_features_seq.device)
        
        output_spikes = [self.decoder(hidden_features_seq[:, t, :]) for t in range(time_steps)]
        rates = torch.stack(output_spikes, dim=1).mean(dim=1).view(batch_size, -1, self.num_attrs)
        
        return {'existence': rates[:, :, 0], 'range': rates[:, :, 1], 'angle': 2 * rates[:, :, 2] - 1, 'velocity': 2 * rates[:, :, 3] - 1}

class SpikingSPHead(nn.Module):
    def __init__(self, hidden_size, range_bins, doppler_bins, **neuron_params):
        super().__init__()
        self.range_bins, self.doppler_bins = range_bins, doppler_bins
        output_neurons = range_bins * doppler_bins
        self.rdm_decoder = ResonateFireLayer(hidden_size, output_neurons, **neuron_params)
        self.cfar_decoder = ResonateFireLayer(hidden_size, output_neurons, **neuron_params)

    def forward(self, hidden_features_seq):
        batch_size, time_steps, _ = hidden_features_seq.shape
        self.rdm_decoder.reset_state(batch_size, hidden_features_seq.device)
        self.cfar_decoder.reset_state(batch_size, hidden_features_seq.device)

        rdm_spikes = [self.rdm_decoder(hidden_features_seq[:, t, :]) for t in range(time_steps)]
        cfar_spikes = [self.cfar_decoder(hidden_features_seq[:, t, :]) for t in range(time_steps)]
        
        rdm_rate = torch.stack(rdm_spikes, dim=1).mean(dim=1)
        cfar_rate = torch.stack(cfar_spikes, dim=1).mean(dim=1)
        
        return {
            'rdm': rdm_rate.view(batch_size, self.range_bins, self.doppler_bins),
            'cfar_logits': cfar_rate.view(batch_size, self.range_bins, self.doppler_bins)
        }

class UnifiedRadarSNN_End2End(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.time_steps = params['time_steps']
        self.num_samples = params['num_samples']
        hidden_size = params['hidden_size']
        num_antennas = params['num_antennas']
        
        neuron_params = {k: v for k, v in params.items() if 'neuron' in k or 'sg' in k}

        conv_out_len = self.num_samples // 4
        self.sconv1 = SpikingConv1d(num_antennas, 32, 7, 2, 3, **neuron_params)
        self.sconv2 = SpikingConv1d(32, 64, 5, 2, 2, **neuron_params)
        self.input_layer = nn.Linear(64 * conv_out_len, hidden_size)
        self.rf_layer1 = ResonateFireLayer(hidden_size, hidden_size, **neuron_params)
        
        # --- FIX: The attention layer was missing here too ---
        # NOTE: For the E2E model, attention on the sequence *before* decoding makes more sense.
        # But this would require a spiking attention mechanism. For simplicity and consistency,
        # we will apply analog attention on the *average* features, then use that to gate the sequence.
        # This is a reasonable architectural choice.
        self.attention = SpatialAttention(hidden_size)
        
        self.od_head = SpikingObjectDetectionHead(hidden_size, params['max_objects'], **neuron_params)
        self.sp_head = SpikingSPHead(hidden_size, params['range_bins'], params['doppler_bins'], **neuron_params)

    def _reset_states(self, batch_size, device):
        conv_out_len = self.num_samples // 4
        self.sconv1.reset_state(batch_size, self.num_samples // 2, device)
        self.sconv2.reset_state(batch_size, conv_out_len, device)
        self.rf_layer1.reset_state(batch_size, device)

    def forward(self, x):
        batch_size, device = x.shape[0], x.device
        self._reset_states(batch_size, device)
        
        hidden_spikes_over_time = []
        for t in range(self.time_steps):
            x_t = x.permute(0, 1, 3, 2)[:, t, :, :]
            s_c1 = self.sconv1(x_t)
            s_c2 = self.sconv2(s_c1)
            s_c2_flat = s_c2.view(batch_size, -1)
            s_in = F.relu(self.input_layer(s_c2_flat))
            s_r1 = self.rf_layer1(s_in)
            hidden_spikes_over_time.append(s_r1)
        
        hidden_features_seq = torch.stack(hidden_spikes_over_time, dim=1)
        
        # --- FIX: Applying attention ---
        # A simple way to apply analog attention to a sequence for a spiking decoder:
        # 1. Calculate average features (rate).
        # 2. Get attention scores from the average.
        # 3. Broadcast these scores back across the time dimension.
        avg_hidden_features = hidden_features_seq.mean(dim=1)
        attention_scores = self.attention.attention_net(avg_hidden_features) # Using inner net
        # Unsqueeze to allow broadcasting: (B, F) -> (B, 1, F)
        attended_features_seq = hidden_features_seq * attention_scores.unsqueeze(1)
        
        # The spiking decoders now process the attended sequence
        od_outputs = self.od_head(attended_features_seq)
        sp_outputs = self.sp_head(attended_features_seq)
        
        return {'sp_outputs': sp_outputs, 'od_outputs': od_outputs}