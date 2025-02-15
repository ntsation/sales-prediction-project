import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .base_model import ModelInterface


class SalesPredictor(ModelInterface):
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        print("SalesPredictor model initialized!")

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("SalesPredictor model trained!")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        print(f"SalesPredictor model evaluated! MAE: {mae}, MSE: {mse}")
        return mae, mse

    def save_model(self, path="models/sales_predictor.pkl"):
        joblib.dump(self.model, path)
        print(f"SalesPredictor model saved!")
