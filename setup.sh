#!/bin/bash

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python and try again."
    exit 1
fi

# Check if virtualenv is installed
if ! command -v virtualenv &> /dev/null; then
    echo "virtualenv is not installed. Installing virtualenv..."
    pip install virtualenv
fi

# Create a virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    virtualenv .env
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .env/bin/activate

# Install the required packages
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

echo "Project setup completed successfully!"
