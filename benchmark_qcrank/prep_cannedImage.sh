#!/bin/bash

# Declare associative array for tag->nqAddr mapping
declare -A tag_qubits=(
    ["a1"]="5"    # 32x1 = 32 pixels ≈ 2^5
    ["b1"]="9"   # 64x16 = 1024 pixels ≈ 2^9
    ["b2"]="9"    # 32x32 = 1024 pixels ≈ 2^9
    ["b3"]="10"   # 64x64 = 4096 pixels ≈ 2^9
    ["b4"]="11"    # 128x128 = 16384 pixels ≈ 2^9
    ["c1"]="10"   # 64x80 = 5120 pixels ≈ 2^10
    ["d1"]="12"   # 192x128 = 24576 pixels ≈ 2^12
    ["d2"]="12"   # 192x128 = 24576 pixels ≈ 2^12
    ["d3"]="12"   # 128x192 = 24576 pixels ≈ 2^12
    ["d4"]="12"   # 192x128 = 24576 pixels ≈ 2^12
    ["d5"]="12"   # 192x128 = 24576 pixels ≈ 2^12
    ["e1"]="15"   # 384x256 = 98304 pixels ≈ 2^15
    ["e2"]="15"   # 256x384 = 98304 pixels ≈ 2^15
    ["e3"]="15"   # 384x256 = 98304 pixels ≈ 2^15
)

# Output directory - matching the default in prep_cannedImage.py
outdir="out"

# Create output directory if it doesn't exist
mkdir -p "$outdir"

# Process each tag
for tag in "${!tag_qubits[@]}"; do
    echo "Processing tag: $tag with nqAddr: ${tag_qubits[$tag]}"
    python3 prep_cannedImage.py --tag "$tag" --nqAddr "${tag_qubits[$tag]}"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $tag"
    else
        echo "✗ Failed to process $tag"
    fi
    echo "------------------------"
done

echo "All images processed!"