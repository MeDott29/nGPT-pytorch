#!/bin/bash

echo "nGPT Explorer - Interactive Learning Suite"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in your PATH."
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

# Check if required packages are installed
echo "Checking required packages..."
python3 check_requirements.py
if [ $? -ne 0 ]; then
    echo "Failed to check requirements."
    echo "Please install the required packages manually:"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Test controller connectivity
echo
echo "Testing Xbox controller connectivity..."
echo "(If no controller is connected, you can use keyboard controls instead)"
python3 test_controller.py

# Run the explorer
echo
echo "Starting nGPT Explorer..."
python3 ngpt_explorer.py 