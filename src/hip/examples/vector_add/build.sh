
#!/bin/bash
# Build script for the vector addition example

set -e  # Exit on error

# Directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo Script directory: $SCRIPT_DIR

# Compile the kernel to a binary file
hipcc -fno-gpu-rdc -fPIC --genco -O3 -o "$SCRIPT_DIR/vector_add.hsaco" "$SCRIPT_DIR/kernel.hip"

echo "Kernel compiled successfully to vector_add.hsaco"