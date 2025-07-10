# SNN for Radar Signal Processing and Object Detection

This project implements a Spiking Neural Network (SNN) model that performs both signal processing and object detection directly from raw FMCW radar data. The model learns to generate a Range-Doppler Map (RDM) and simultaneously detect objects (cars, persons, cyclists) with their properties (range, angle, etc.).

## Features

- **End-to-End Learning:** Processes raw ADC samples to produce high-level detections.
- **Feature Extraction:** A convolutional SNN maps range, doppler and angle, RDM as feature vector
- **Hidden Architecture:** A fully connected SNN learns the pattern from the extracted feature vector 
- **Output Layer:** A GRU (Gated Recurrent Unit) decode the learned pattern and feeds two separate heads for signal processing and object detection.
- **Bio-inspired SNN:** Uses Resonate-and-Fire neurons with surrogate gradients for energy-efficient, event-based processing.
- **Modular Codebase:** The source code is organized into logical modules for clarity and extensibility.

## Project Structure

snn-radar-unified/
  - results/ # Outputs are saved here
  - src/ # Main source code
  - test.py # Seperate testing module
  - grid_search_snn/cnn # To perform grid search
  - requirements.txt # Dependencies
  - README.md # This file

## How to run the script

From root directory, on the terminal execute the following command: python -m src.main --model_type cnn/cnn_mlp/snn_hybrid/snn_e2e
