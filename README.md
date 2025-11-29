# 🔒 Privacy-Preserving Credit Default Risk Prediction with FHE

A complete end-to-end application demonstrating Fully Homomorphic Encryption (FHE) for secure machine learning inference. Your financial data is encrypted and never seen by the server!

## Project Overview

This application allows users to predict credit card default risk while keeping their financial data completely private using Microsoft's TenSEAL library. The model predicts whether a customer will default on their credit card payment (binary classification: default vs no default).

**Key Features:**
- 🔐 **Client-side encryption** - Data encrypted before sending
- ⚡ **Encrypted processing** - Server never sees raw data
- 🎨 **Beautiful UI** - Streamlit-based interface
- 🚀 **Fast API** - High-performance backend
- 🐳 **Docker ready** - Easy deployment

## 🛠️ Tech Stack

- **Backend**: FastAPI + TenSEAL (Microsoft SEAL)
- **Frontend**: Streamlit
- **ML**: scikit-learn + joblib
- **FHE**: TenSEAL (CKKS scheme)
- **Deployment**: Docker + Hugging Face Spaces

## 📁 Project Structure

```
fhe-default-risk-prediction/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── fhe_model.py         # FHE model implementation
│   ├── requirements.txt     # Backend dependencies
│   └── models/              # Trained models (generated)
│       ├── credit_model.pkl
│       ├── logistic_regression.pkl
│       ├── random_forest.pkl
│       ├── gradient_boosting.pkl
│       ├── scaler.pkl
│       └── metadata files
├── frontend/
│   ├── streamlit_app.py     # Streamlit UI
│   ├── demo_data.py         # Demo test cases
│   └── requirements.txt     # Frontend dependencies
├── data/
│   └── credit_card_default.csv  # UCI dataset (30,000 records)
├── train_credit_model.py    # Model training script
├── setup_and_run.sh         # Automated setup & run
├── start_backend.sh         # Start backend server
├── start_frontend.sh        # Start frontend UI
└── requirements.txt         # Main dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### Setup

##  OPTION 1: Automated Setup (Recommended)

**Run everything with one script!**

```bash
cd fhe-default-risk-prediction

# Make script executable (first time only)
chmod +x setup_and_run.sh

# Run the complete setup
./setup_and_run.sh
```

**What this script does:**
1. ✅ Creates virtual environment
2. ✅ Installs all dependencies
3. ✅ Trains the ML model (if not already trained)
4. ✅ Starts backend server
5. ✅ Starts frontend UI

**Access the Application:**
- **Frontend UI**: http://localhost:8501
- **API Health**: http://localhost:8000/
---

##  OPTION 2: Manual Setup (Step-by-Step)

### Step 1: Setup Environment

```bash
cd fhe-default-risk-prediction

# Create virtual environment
python3 -m venv fhe-env

# Activate environment
# On macOS/Linux:
source fhe-env/bin/activate
# On Windows:
fhe-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
# Train the credit default prediction model
python train_credit_model.py
```

**Output:** Trained models saved in `backend/models/`

### Step 3: Start Backend Server

```bash
# Open Terminal 1
cd backend
python main.py
```

Backend will run on: http://localhost:8000

### Step 4: Start Frontend UI

```bash
# Open Terminal 2 (new terminal)
cd frontend
streamlit run streamlit_app.py
```

Frontend will run on: http://localhost:8501

### Step 5: Access the Application

- **Frontend UI**: http://localhost:8501
- **API Health Check**: http://localhost:8000/

---

## Stopping the Application

**If using automated setup:**
```bash
# Press Ctrl+C in the terminal to stop both services
```

**If using manual setup:**
```bash
# Press Ctrl+C in each terminal (backend and frontend)
```

## 📊 How It Works

1. **Data Input**: User enters financial information in the web interface
2. **Client Encryption**: Data is encrypted using TenSEAL on the client side
3. **Encrypted Processing**: Server performs ML inference on encrypted data
4. **Secure Results**: Only the client can decrypt the final prediction
5. **Privacy Guaranteed**: Server never sees raw financial data

## 🔐 Security Features

- **Fully Homomorphic Encryption**: Computations on encrypted data
- **Client-side Encryption**: Data encrypted before transmission
- **No Data Storage**: Server doesn't store user data
- **Secure Communication**: All data encrypted in transit

## 📈 Performance Notes

- **FHE Operations**: Slower than regular ML (2-5 seconds per prediction)
- **Model Size**: Optimized for FHE constraints
- **Memory Usage**: Higher due to encryption overhead
- **Scalability**: Suitable for demo/prototype use

## 📝 What to Expect

Once running, you can:
1. 🎯 **Try Demo Cases** - Pre-loaded customer profiles
2. 🔐 **Test FHE Encryption** - See encrypted vs clear predictions
3. 📊 **Compare Models** - Benchmark Logistic Regression vs Random Forest vs Gradient Boosting
4. 📈 **View Performance Metrics** - Accuracy, F1 Score, ROC AUC

## 🧪 Testing the API

You can test the API directly using curl:

```bash
# Health check
curl http://localhost:8000/

# Get FHE context
curl http://localhost:8000/context

# Test prediction (23 features required)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [20000, 2, 2, 1, 24, 2, 2, -1, -1, -2, -2, 3913, 3102, 689, 0, 0, 0, 0, 689, 0, 0, 0, 0]}'
```


## 🔍 Troubleshooting

**Issue: Port already in use**
```bash
# Kill process on port 8000 (backend)
lsof -ti:8000 | xargs kill -9

# Kill process on port 8501 (frontend)
lsof -ti:8501 | xargs kill -9
```

**Issue: Module not found**
```bash
# Make sure virtual environment is activated
source fhe-env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Issue: Model not found**
```bash
# Train the model first
python train_credit_model.py
```

## 📚 Documentation

- **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)** - Detailed technical guide explaining:
  - How models are trained
  - How data is encrypted
  - How inference works
  - How libraries are used
  - Complete workflow examples

- **[api-documentation.yaml](api-documentation.yaml)** - OpenAPI 3.0 specification:
  - Complete API endpoint documentation
  - Request/response schemas
  - Example requests and responses
  - Interactive Swagger UI at http://localhost:8000/docs

## 🔗 External Resources

- [TenSEAL Documentation](https://github.com/OpenMined/TenSEAL)
- [Microsoft SEAL](https://github.com/Microsoft/SEAL)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
