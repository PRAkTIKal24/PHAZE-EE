#!/bin/bash

# JetClass Dataset Download Script
# Dataset: https://zenodo.org/records/6619768
# Total size: ~187GB download, ~225GB extracted for full dataset
# This script downloads train/val/test splits

set -e

# Default download directory
DOWNLOAD_DIR="${1:-/workspaces/PHAZE-EE/data/jetclass}"

echo "JetClass Dataset Download"
echo "========================="
echo "Download directory: $DOWNLOAD_DIR"
echo ""
echo "WARNING: The full JetClass dataset requires ~187GB download + ~225GB extracted"
echo "You may want to download a subset for testing purposes."
echo ""

# Create directory structure
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# Zenodo record ID for JetClass
ZENODO_RECORD="6619768"
BASE_URL="https://zenodo.org/records/${ZENODO_RECORD}/files"

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
echo "1) Light download - validation set only"
echo "   Download: ~7.5GB | Extracted: ~9GB"
echo ""
echo "2) Paper download - train files 0-4, val, and test sets for paper reproduction"
echo "   Download: ~112GB | Extracted: ~135GB"
echo ""
echo "3) Full dataset - all train files (0-9), val, and test sets"
echo "   Download: ~187GB | Extracted: ~225GB"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Downloading light subset (validation set only)..."
        
        # Download validation set
        download_file "JetClass_Pythia_val_5M.tar"
        extract_file "JetClass_Pythia_val_5M.tar" "val"
        
        echo ""
        echo "Validation set downloaded to: $DOWNLOAD_DIR/val"
        echo "For testing, you can use a subset of these files in your training config."
        ;;
    2)
        echo "Downloading paper subset (train 0-4, val, test)..."
        
        # Training set files 0-4
        for i in {0..4}; do
            download_file "JetClass_Pythia_train_100M_part${i}.tar"
            extract_file "JetClass_Pythia_train_100M_part${i}.tar" "train"
        done
        
        # Validation set
        download_file "JetClass_Pythia_val_5M.tar"
        extract_file "JetClass_Pythia_val_5M.tar" "val"
        
        # Test set
        download_file "JetClass_Pythia_test_20M.tar"
        extract_file "JetClass_Pythia_test_20M.tar" "test"
        
        echo ""
        echo "Paper subset downloaded to: $DOWNLOAD_DIR"
        echo "Train files 0-4, validation, and test sets are ready."
        ;;
    3)
        echo "Downloading full dataset (all train files, val, test)..."
        echo "This will take a considerable amount of time and storage."
        
        # Training set (all 10 files)
        for i in {0..9}; do
            download_file "JetClass_Pythia_train_100M_part${i}.tar"
            extract_file "JetClass_Pythia_train_100M_part${i}.tar" "train"
        done
        
        # Validation set
        download_file "JetClass_Pythia_val_5M.tar"
        extract_file "JetClass_Pythia_val_5M.tar" "val"
        
        # Test set
        download_file "JetClass_Pythia_test_20M.tar"
        extract_file "JetClass_Pythia_test_20M.tar" "test"
        
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
