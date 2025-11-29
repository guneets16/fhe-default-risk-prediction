#!/bin/bash

echo " Starting FHE Credit Default Risk Prediction Frontend..."
echo ""

# Activate virtual environment
source fhe-env/bin/activate

# Start frontend
cd frontend
streamlit run streamlit_app.py



