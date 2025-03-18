#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download required NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Setup complete. Activate the virtual environment with 'source venv/bin/activate'"
echo "Run the application with 'python src/main.py \"What is the impact of AI?\"'"
