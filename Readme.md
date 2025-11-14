
# 🚀 Mobile Payments Real-Time Fraud Analytics

This project is an end-to-end **fraud detection system** built for mobile payments.
It covers everything from **data processing and model training** to a **real-time FastAPI service** that scores transactions instantly.

I built this project to simulate a real-world workflow used in payment gateways, wallets, neo-banks, and fintech fraud teams.

---

## 🌟 What This Project Does

- Detects **fraudulent mobile payment transactions** in real-time
- Uses **XGBoost + SMOTE** to handle highly imbalanced fraud data
- Builds meaningful fraud features (time, device, category, amount patterns)
- Serves predictions through a **FastAPI** endpoint
- Uses **Docker** for easy deployment
- Supports both **single transaction** and **bulk CSV** predictions
- Stores all prediction logs
- Ensures consistent encodings by saving label encoders

It’s essentially a small version of what Razorpay, Paytm, Stripe, or PhonePe would use in production.

---

## 🧠 Tech Used

- **Python**
- **XGBoost / SMOTE / Scikit-Learn**
- **FastAPI**
- **Pandas / NumPy**
- **Joblib**
- **Docker**

---

## 📂 Project Structure

```
Mobile_Payments_Real_time_Fraud_Analytics/
│
├── data/                    # training data & prediction logs
├── model/                   # saved model + encoders
├── src/                     # training + API code
│   ├── pipeline.py
│   ├── train_model.py
│   ├── serve.py
│
├── docker/                  # Dockerfile for deployment
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1️⃣ Install requirements

```
pip install -r requirements.txt
```

### 2️⃣ Train the model

```
python src/train_model.py
```

This will:

- Clean data
- Generate fraud features
- Handle imbalance with SMOTE
- Train an XGBoost model
- Save model + encoders into `/model`

### 3️⃣ Start the API

```
uvicorn src.serve:app --reload
```

API docs appear at:

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Example Requests (Postman or Curl)

### 🔹 Single Transaction

```
POST /predict
```

```json
{
  "transaction_id": "TXN000028",
  "timestamp": "2025-05-19 23:50:00",
  "user_id": "1008",
  "amount": 11872.71,
  "merchant_id": "2002",
  "location": "Bangalore",
  "device_type": "Android",
  "transaction_type": "top-up"
}
```

The API returns:

- `fraud_probability`
- `predicted_label (0/1)`
- `predicted_at`
- Full transaction details

### 🔹 Bulk CSV

```
POST /predict_csv
```

Upload any CSV containing the required fields.

---

## 🐳 Running with Docker

Build:

```
docker build -t fraud-api -f docker/Dockerfile .
```

Run:

```
docker run -p 8000:8000 fraud-api
```

---

## 📊 Model Performance

Typical results (dependent on dataset):

- **ROC-AUC:** ~0.90+
- **PR-AUC:** Strong for fraud detection
- Captures patterns like:

  - Unusual device
  - High amounts
  - Night-time transactions
  - New merchants
  - Suspicious transaction types

---

## 🎯 Why I Built This

I wanted a project that feels **real**, not just a Jupyter notebook.
This codebase shows:

- Practical ML engineering
- Serving ML models with FastAPI
- Handling imbalanced fraud datasets (real-world issue)
- Feature engineering for fraud analytics
- Building production-ready pipelines
- Packaging an ML app with Docker

This is the kind of ML system used by fintech companies handling millions of transactions per day.

---

## ✨ Author

**Virender Machra**
Machine Learning & Generative AI Engineer
Working on fraud systems, NLP, data pipelines, and LLM applications.

