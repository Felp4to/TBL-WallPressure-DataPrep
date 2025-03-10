import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class XGBoostModel:
    def __init__(self, folder, params=None, num_boost_round=100, early_stopping_rounds=10, window=20):
        self.params = params if params else {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'rmse',
            'objective': "reg:squarederror"
        }

        self.model = xgb.XGBRegressor(**self.params, enable_categorical=False)
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.window = window
        self.history = None
        self.folder = os.path.join("./tests", folder)
        self.create_folder()
    
    def create_folder(self):
        os.makedirs(self.folder, exist_ok=True)
    
    def train(self, X_train, y_train, X_val, y_val):
        eval_set = [(X_train, y_train), (X_val, y_val)]
        start_time = time.time()

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            #early_stopping_rounds=self.early_stopping_rounds,          
            verbose=True
        )

        training_time = time.time() - start_time
        self.history = self.model.evals_result()

        # Salva il tempo di addestramento
        time_path = os.path.join(self.folder, "training_time.txt")
        with open(time_path, "w") as f:
            f.write(f"Tempo totale di addestramento: {training_time:.2f} secondi\n")
        print(f"Tempo totale di addestramento salvato in {time_path}")

        # Salva la cronologia dell'addestramento
        history_path = os.path.join(self.folder, "training_history.csv")
        df_history = pd.DataFrame(self.history['validation_0'])
        df_history.to_csv(history_path, index=False)
        print(f"Cronologia di addestramento salvata in {history_path}")

        # Plot delle metriche di addestramento
        self.plot_training_metrics()

    def predict(self, x_test):
        """Genera previsioni."""
        return self.model.predict(x_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
        results_path = os.path.join(self.folder, "evaluation_metrics.csv")
        pd.DataFrame([results]).to_csv(results_path, index=False)
        print(f"Metriche salvate in {results_path}")
        return results
    
    def save_model(self):
        model_path = os.path.join(self.folder, "xgboost_model.json")
        self.model.save_model(model_path)
        print(f"Modello salvato in {model_path}")
    
    def load_model(self):
        model_path = os.path.join(self.folder, "xgboost_model.json")
        if os.path.exists(model_path):
            self.model.load_model(model_path)
            print(f"Modello caricato da {model_path}")
        else:
            print(f"Nessun modello trovato in {model_path}.")

    def compare_predictions(self, y_test, y_pred):
        """Confronta i valori reali con le predizioni e li salva in un CSV."""
        comparison_df = pd.DataFrame({
            "Valore Reale": y_test.flatten(),
            "Predizione": y_pred.flatten()
        })

        comparison_path = os.path.join(self.folder, "predictions_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)

        print(f"Confronto predizioni salvato in {comparison_path}")
        return comparison_df
    
    def plot_training_metrics(self):
        if self.history is None:
            print("Nessuna cronologia di addestramento disponibile.")
            return
        
        plt.figure(figsize=(10, 5))
        for metric in self.history['validation_0'].keys():
            plt.plot(self.history['validation_0'][metric], label=f"Training {metric}")
            plt.plot(self.history['validation_1'][metric], label=f"Validation {metric}")
        
        plt.xlabel("Boosting Rounds")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.title("Training and Validation Metrics Over Time")
        plt.grid()
        plt.savefig(os.path.join(self.folder, "training_metrics.png"))
        plt.show()
        print("Grafico delle metriche di training salvato.")
















