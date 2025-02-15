# Predictive Analysis for Sales Forecasting

This project uses **Machine Learning** to predict future sales based on historical data. It implements a complete pipeline, including **preprocessing, model training, and deployment via FastAPI**.

---

## Project Structure
```
sales-prediction-project/
│── data/                # Stores the dataset
│── models/              # Trained models
│── src/                 # Main project code
│   ├── api/             # API code (FastAPI)
│   ├── ml/              # Model training code
│   ├── utils/           # Utility functions
│── tests/               # Unit tests
│── Makefile             # Command automation
│── requirements.txt     # Project dependencies
│── README.md            # Documentation
```

---

## Technologies Used

- **Python**
- **Pandas** & **Scikit-Learn**
- **MLflow** (Experiment Tracking)
- **FastAPI** (Inference API)
- **SOLID** (Best practices in code architecture)
- **Makefile** (Task automation)

---

## Useful Commands
| Command       | Description |
|--------------|-------------|
| `make init`   | Sets up the virtual environment and installs dependencies. |
| `make train`  | Downloads the dataset, trains the model, and saves logs. |
| `make backend` | Starts the FastAPI backend for predictions. |
| `make clean` | Removes temporary files and the virtual environment. |

---

## Setup

### 1. Set up the environment
```bash
make init
```
This will create a virtual environment and install all dependencies.

### 2. Download the Dataset and Train the Model
```bash
make train
```
What happens?  
- Trains a regression model for sales forecasting.
- Saves the trained model in the `models/` directory.

### 3. Run the API for Inference
```bash
make backend
```
The API will be available at:  
http://127.0.0.1:8000

You can test the API in Swagger UI:  
http://127.0.0.1:8000/docs

---

## Accessing MLflow
To visualize the training experiments:

```bash
mlflow ui
```
Access it in your browser:  
http://127.0.0.1:5000

If running on a remote machine:  
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

---

## API Usage Example
### Making a Prediction

Parameters:
| Field       | Description |
|------------|-------------|
| `1.705833e+09` | Transaction timestamp |
| `0` | Payment type (e.g., cash/card) |
| `12` | Number of products in the cart |
| `4` | Type of product purchased |

Endpoint: `POST /predict`

Example request using `cURL`:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features":  [1.705833e+09, 0, 12, 4]
}'
```

Input JSON:
```json
{
  "features": [1.705833e+09, 0, 12, 4]
}
```

Example Response:
```json
{
  "prediction": 200.45
}
```
```
