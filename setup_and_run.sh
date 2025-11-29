#!/bin/bash

echo "FHE - PPML - Setup and Run"
echo "======================================"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "fhe-env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv fhe-env
    echo "Virtual environment created!"
    echo ""
fi

# Activate virtual environment
echo " Activating virtual environment..."
source fhe-env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "Dependencies installed!"
echo ""

# Train model
echo "Training model from scratch..."
python3 train_credit_model.py

echo ""
echo "======================================"
echo " Setup complete!"
echo "======================================"
echo ""
echo " Starting servers..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT SIGTERM

# Start backend server in background
echo "Starting backend server on http://localhost:8000..."
cd backend
../fhe-env/bin/uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend server in background
echo "Starting frontend UI on http://localhost:8501..."
cd frontend
../fhe-env/bin/streamlit run streamlit_app.py &
FRONTEND_PID=$!
cd ..

echo ""
echo "======================================"
echo "Application is running!"
echo "======================================"
echo ""
echo "📱 Frontend UI: http://localhost:8501"
echo "🔌 Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for both processes
wait

