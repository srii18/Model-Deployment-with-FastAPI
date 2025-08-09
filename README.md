# 🌸 Iris Species Prediction API

A FastAPI-based machine learning service that predicts iris flower species from petal/sepal measurements using a trained Random Forest model.  
This project is fully containerized with Docker for easy deployment.

---

## 📦 Features
- **Single Prediction** (`/predict`) — Predict the species for one flower.
- **Batch Prediction** (`/predict-batch`) — Predict multiple flowers in one request.
- **Health Check** (`/health`) — Ensure the API and model are running correctly.
- **Model Info** (`/model-info`) — Get details about the trained model and feature importances.
- Interactive **Swagger UI** at `/docs`.

---

## 🚀 Getting Started

### 1. Build Docker Image
Run from the folder containing your `Dockerfile`:
```bash
docker build -t fastapi-iris:latest .
```

### 2. Run Container
```bash
docker run --rm -p 8000:8000 fastapi-iris:latest
```

---

## 🌐 Access the API
Once the container is running:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)  
- ReDoc UI: [http://localhost:8000/redoc](http://localhost:8000/redoc)  
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)  

---

## 🧪 Example Requests

### Health Check
```bash
curl http://localhost:8000/health
```
**Response**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "test_prediction": "setosa"
}
```

---

### Single Prediction
```bash
curl --request POST   --url http://localhost:8000/predict   --header 'Content-Type: application/json'   --data '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}'
```
**Response**
```json
{
  "species": "setosa",
  "confidence": 0.99,
  "probabilities": {
    "setosa": 0.99,
    "versicolor": 0.01,
    "virginica": 0.0
  },
  "input_features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

---

### Batch Prediction
```bash
curl --request POST   --url http://localhost:8000/predict-batch   --header 'Content-Type: application/json'   --data '{
    "samples": [
      {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
      {"sepal_length": 6.5, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
      {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5}
    ]
}'
```
**Response**
```json
{
  "predictions": [
    {
      "species": "setosa",
      "confidence": 1,
      "probabilities": {"setosa": 1, "versicolor": 0, "virginica": 0},
      "input_features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    },
    {
      "species": "setosa",
      "confidence": 0.8822105589446015,
      "probabilities": {"setosa": 0.8822105589446015, "versicolor": 0.10410274393941478, "virginica": 0.013686697115983724},
      "input_features": {"sepal_length": 6.5, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    },
    {
      "species": "versicolor",
      "confidence": 0.9575311841687991,
      "probabilities": {"setosa": 0.0030000326288120243, "versicolor": 0.9575311841687991, "virginica": 0.03946878320238887},
      "input_features": {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5}
    }
  ],
  "total_samples": 3
}
```

---

## 🛠 Tech Stack
- **Python 3.11**
- **FastAPI** — Web framework
- **scikit-learn** — Machine learning
- **Uvicorn** — ASGI server
- **Docker** — Containerization

---

## 📜 License
This project is for educational/demo purposes.

---

## ✨ Author
Developed as part of the **AI Engineering Internship Assignment** — Model Deployment with FastAPI.
