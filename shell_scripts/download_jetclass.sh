#!/bin/bash

# JetClass Dataset Download Script
# Dataset: https://zenodo.org/record/6619768
# Total size: ~1.2 TB for full dataset
# This script downloads train/val/test splits

set -e

# Default download directory
DOWNLOAD_DIR="${1:-/workspaces/PHAZE-EE/data/jetclass}"

echo "JetClass Dataset Download"
echo "========================="
echo "Download directory: $DOWNLOAD_DIR"
echo ""
echo "WARNING: The full JetClass dataset is ~1.2 TB"
echo "You may want to download a subset for testing purposes."
echo ""

# Create directory structure
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# Zenodo record ID for JetClass
ZENODO_RECORD="6619768"
BASE_URL="https://zenodo.org/record/${ZENODO_RECORD}/files"

# Function to download a file
download_file() {
    local filename=$1
    local url="${BASE_URL}/${filename}"
    
    if [ -f "$filename" ]; then
        echo "File $filename already exists, skipping..."
    else
        echo "Downloading $filename..."
        wget --no-check-certificate "$url" -O "$filename"
    fi
}

# Function to extract a file
extract_file() {
    local filename=$1
    local extract_dir=$2
    
    echo "Extracting $filename to $extract_dir..."
    mkdir -p "$extract_dir"
    tar -xzf "$filename" -C "$extract_dir"
}

echo "Select download option:"
echo "1) Download sample subset (~10GB) - recommended for testing"
echo "2) Download validation set only (~50GB)"
echo "3) Download full dataset (~1.2TB) - requires significant storage"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Downloading sample subset..."
        echo "NOTE: Creating a 1% sample from validation set for quick testing"
        
        # Download validation set
        download_file "JetClass_val.tar.gz"
        extract_file "JetClass_val.tar.gz" "val"
        
        echo ""
        echo "Sample subset downloaded to: $DOWNLOAD_DIR/val"
        echo "For testing, you can use a subset of these files in your training config."
        ;;
    2)
        echo "Downloading validation set..."
        download_file "JetClass_val.tar.gz"
        extract_file "JetClass_val.tar.gz" "val"
        
        echo ""
        echo "Validation set downloaded to: $DOWNLOAD_DIR/val"
        ;;
    3)
        echo "Downloading full dataset (train + val + test)..."
        echo "This will take a considerable amount of time and storage."
        
        # Training set (largest)
        download_file "JetClass_train.tar.gz"
        extract_file "JetClass_train.tar.gz" "train"
        
        # Validation set
        download_file "JetClass_val.tar.gz"
        extract_file "JetClass_val.tar.gz" "val"
        
        # Test set
        download_file "JetClass_test.tar.gz"
        extract_file "JetClass_test.tar.gz" "test"
        
        echo ""
        echo "Full dataset downloaded to: $DOWNLOAD_DIR"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Download complete!"
echo ""
echo "Next steps:"
echo "1. Update your project config to point to the data files:"
echo "   train_files: glob('$DOWNLOAD_DIR/train/**/*.root')"
echo "   val_files: glob('$DOWNLOAD_DIR/val/**/*.root')"
echo "   test_files: glob('$DOWNLOAD_DIR/test/**/*.root')"
echo ""
echo "2. Make sure your data_config points to: data_configs/JetClass_full.yaml"
