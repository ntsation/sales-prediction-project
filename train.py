import mlflow
import mlflow.sklearn
from src.utils.data_loader import load_data
from src.models.sales_model import SalesPredictor
import logging

mlflow.set_experiment("sales-prediction")

logging.info("Starting training!")

with mlflow.start_run():
    X_train, X_test, y_train, y_test = load_data(filepath="data/sales_data.csv")

    model = SalesPredictor()
    model.train(X_train, y_train)

    mae, mse = model.evaluate(X_test, y_test)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)

    model.save_model()

logging.info("Model trained and saved in MLflow!")
