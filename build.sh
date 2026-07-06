#!/usr/bin/env bash
# Render runs this script once during deployment to train and save the model.
# This ensures model.pkl, label_encoders.pkl, and feature_names.pkl exist
# before the Flask app starts.

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Training model and saving artefacts..."
python train_model.py

echo "Build complete."