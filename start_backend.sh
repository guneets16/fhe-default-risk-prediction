#!/bin/bash

echo " Starting FHE Credit Default Risk Prediction Backend..."
echo ""

# Activate virtual environment
source fhe-env/bin/activate

# Start backend
cd backend
uvicorn main:app --reload --port 8000

