import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RandomForestModel:
    def __init__(self, folder, n_estimators=100, max_depth=None, random_state=42, window=20):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.window = window
        self.folder = os.path.join("./tests", folder)
        self.create_folder()
        self.save_params()
    
    def create_folder(self):
        os.makedirs(self.folder, exist_ok=True)

    def save_params(self):
        params_path = os.path.join(self.folder, "model_params.json")
        with open(params_path, "w") as f:
            json.dump(self.params, f, indent=4)
        print(f"Parametri del modello salvati in {params_path}")
        
        self.save_model()

    def train(self, X_train, y_train, X_val, y_val):
        start_time = time.time()
        
        self.model.fit(X_train, y_train.ravel())
        
        val_errors = []
        train_sizes = np.linspace(0.1, 1.0, 100)  # Simuliamo 100 step di training progressivo
        
        for size in train_sizes:
            subset_size = int(size * len(X_train))
            self.model.fit(X_train[:subset_size], y_train[:subset_size].ravel())
            y_val_pred = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_errors.append(val_mse)
        
        self.save_training_errors(train_sizes, val_errors)
        
        training_time = time.time() - start_time
        
        time_path = os.path.join(self.folder, "training_time.txt")
        with open(time_path, "w") as f:
            f.write(f"Tempo totale di addestramento: {training_time:.2f} secondi\n")
        print(f"Tempo totale di addestramento salvato in {time_path}")
        
        self.plot_training_curve(train_sizes, val_errors)
        self.save_model()
        
    def save_training_errors(self, train_sizes, val_errors):
        errors_df = pd.DataFrame({"Training Size": train_sizes * len(train_sizes), "Validation MSE": val_errors})
        errors_path = os.path.join(self.folder, "training_errors.csv")
        errors_df.to_csv(errors_path, index=False)
        print(f"Andamento dell'errore salvato in {errors_path}")

    def plot_training_curve(self, train_sizes, val_errors):
        plt.figure(figsize=(10, 5))
        plt.plot(train_sizes * len(train_sizes), val_errors, marker='o', linestyle='--', color='r', label='Validation MSE')
        plt.xlabel("Numero di campioni di training")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title("Andamento dell'errore di validazione")
        plt.legend()
        plt.grid()
        plot_path = os.path.join(self.folder, "training_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Grafico dell'andamento dell'errore di validazione salvato in {plot_path}")
    
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
        results_path = os.path.join(self.folder, "evaluation_metrics_test_set.csv")
        pd.DataFrame([results]).to_csv(results_path, index=False)
        print(f"Metriche salvate in {results_path}")
        return results
    
    def save_model(self):
        model_path = os.path.join(self.folder, "random_forest_model.pkl")
        pd.to_pickle(self.model, model_path)
        print(f"Modello salvato in {model_path}")
    
    def load_model(self):
        model_path = os.path.join(self.folder, "random_forest_model.pkl")
        if os.path.exists(model_path):
            self.model = pd.read_pickle(model_path)
            print(f"Modello caricato da {model_path}")
        else:
            print(f"Nessun modello trovato in {model_path}.")
    
    def compare_predictions(self, y_test, y_pred):
        comparison_df = pd.DataFrame({
            "Valore Reale": y_test.flatten(),
            "Predizione": y_pred.flatten()
        })

        comparison_path = os.path.join(self.folder, "predictions_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Confronto predizioni salvato in {comparison_path}")
        return comparison_df

    def plot_predictions(self, y_test, y_pred, slice=50):
        plt.figure(figsize=(15, 8))
        plt.plot(y_test[:slice], label='Valori Reali', color='blue')
        plt.plot(y_pred[:slice], label='Predizioni', color='red', linestyle='dashed')
        plt.legend()
        plt.title('Confronto tra Predizioni e Valori Reali')
        plot_path = os.path.join(self.folder, "predictions_plot_1.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Grafico salvato in {plot_path}")

        plt.figure(figsize=(15, 8))
        plt.plot(y_test, label='Valori Reali', color='blue')
        plt.plot(y_pred, label='Predizioni', color='red', linestyle='dashed')
        plt.legend()
        plt.title('Confronto tra Predizioni e Valori Reali')
        plot_path = os.path.join(self.folder, "predictions_plot_tot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Grafico salvato in {plot_path}")