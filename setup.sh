#!/bin/bash

# Activate conda shell hook 
eval "$(conda shell.bash hook)"

# Check conda environment
if ! conda info --envs | grep -q "^figma-env "; then
    echo "🔹 Creating Conda environment 'figma-env'..."
    conda create -y --name figma-env python=3.8
else
    echo "✅ Conda environment 'figma-env' already exists."
fi

# Activate conda environment
echo "🔹 Activating Conda environment..."
conda activate figma-env

# Check conda envirioment activation
if [[ $(which python) != *"figma-env"* ]]; then
    echo "❌ Conda environment activation failed. Please check your Conda installation."
    exit 1
fi

# Install package
echo "🔹 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"

