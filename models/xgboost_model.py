import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from joblib import dump, load



class XGBoostModel:
    def __init__(self, folder, n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, patience=5, output_units=1, eval_metrics=['rmse', 'mae']):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='reg:squarederror',
            eval_metric=eval_metrics
        )
        
        self.folder = folder
        self.create_folder()
        self.patience = patience
        self.output_units = output_units
        self.history = None

    def create_folder(self):
        os.makedirs(self.folder, exist_ok=True)
    
    def train(self, x_train, y_train, x_val=None, y_val=None):
        start_time = time.time()
        eval_set = [(x_train, y_train)]
        
        if x_val is not None and y_val is not None:
            eval_set.append((x_val, y_val))
            self.model.fit(
                x_train, y_train,
                eval_set=eval_set,
                verbose=True
            )
            self.history = self.model.evals_result()
        else:
            self.model.fit(x_train, y_train)
            self.history = self.model.evals_result()
        
        training_time = time.time() - start_time
        
        time_path = os.path.join(self.folder, "training_time.txt")
        with open(time_path, "w") as f:
            f.write(f"Tempo totale di addestramento: {training_time:.2f} secondi\n")
        print(f"Tempo di addestramento salvato in {time_path}")
        
        self.save_model()

    def save_parameters(self):
        """ Salva i parametri del modello in un file di testo. """
        params_path = os.path.join(self.folder, "model_parameters.txt")

        params = {
            "Folder": self.folder,
            "N Estimators": self.model.get_params()["n_estimators"],
            "Learning Rate": self.model.get_params()["learning_rate"],
            "Max Depth": self.model.get_params()["max_depth"],
            "Subsample": self.model.get_params()["subsample"],
            "Colsample by Tree": self.model.get_params()["colsample_bytree"],
            "Objective": self.model.get_params()["objective"],
            "Evaluation Metrics": ", ".join(self.model.get_params()["eval_metric"]),
            "Patience": self.patience,
            "Output Units": self.output_units
        }

        with open(params_path, "w") as f:
            f.write("Parametri del Modello XGBoost:\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")

        print(f"Parametri salvati in {params_path}")

    def save_training_history_txt(self):
        if self.history is None:
            print("Nessuna history disponibile da salvare.")
            return
        
        history_path = os.path.join(self.folder, "training_history.txt")
        
        with open(history_path, "w") as f:
            f.write("Training History:\n\n")
            metrics = list(self.history['validation_0'].keys())
            num_rounds = len(self.history['validation_0'][metrics[0]])
            
            for i in range(num_rounds):
                line = f"[{i}]"
                for dataset in self.history:
                    for metric in metrics:
                        value = self.history[dataset][metric][i]
                        line += f"\t{dataset}-{metric}:{value:.5f}"
                f.write(line + "\n")
        
        print(f"Training history salvata in {history_path}")
    
    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred.reshape(-1, self.output_units)
    
    def save_model(self):
        model_path = os.path.join(self.folder, "xgboost_model.joblib")
        dump(self.model, model_path)
        print(f"Modello salvato in {model_path}")
    
    def load_model(self):
        model_path = os.path.join(self.folder, "xgboost_model.joblib")
        if os.path.exists(model_path):
            self.model = load(model_path)
            print(f"Modello caricato da {model_path}")
        else:
            print(f"Nessun modello trovato in {model_path}.")
    
    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        
        if y_test.shape != y_pred.shape:
            raise ValueError(f"Errore: la forma di y_test {y_test.shape} e y_pred {y_pred.shape} non corrisponde.")
        
        num_outputs = y_test.shape[1]
        detailed_results = {}
        mean_results = {}
        
        if num_outputs == 1:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            detailed_results = {
                "Output": ["Unico"],
                "MSE": [mse],
                "RMSE": [rmse],
                "MAE": [mae],
                "R2": [r2]
            }
            mean_results = detailed_results
        else:
            detailed_results["Output"] = [f"Output {i+1}" for i in range(num_outputs)]
            detailed_results["MSE"] = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            detailed_results["RMSE"] = [np.sqrt(detailed_results["MSE"][i]) for i in range(num_outputs)]
            detailed_results["MAE"] = [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            detailed_results["R2"] = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            
            mean_results = {
                "Output": ["Media"],
                "MSE": [np.mean(detailed_results["MSE"])],
                "RMSE": [np.sqrt(np.mean(detailed_results["MSE"]))],
                "MAE": [np.mean(detailed_results["MAE"])],
                "R2": [np.mean(detailed_results["R2"])],
            }
        
        detailed_path = os.path.join(self.folder, "evaluation_metrics_detailed.csv")
        pd.DataFrame(detailed_results).to_csv(detailed_path, index=False)
        
        mean_path = os.path.join(self.folder, "evaluation_metrics_mean.csv")
        pd.DataFrame(mean_results).to_csv(mean_path, index=False)
        
        print(f"Metriche dettagliate salvate in: {detailed_path}")
        print(f"Metriche medie salvate in: {mean_path}")
        
        return {"detailed": detailed_results, "mean": mean_results}
    
    def save_training_history(self):
        history_path = os.path.join(self.folder, "training_history.npy")
        np.save(history_path, self.history)
        print(f"Training history salvata in {history_path}")
    
    def compare_predictions(self, y_test, y_pred):
        num_outputs = y_test.shape[1]
        
        if num_outputs == 1:
            comparison_df = pd.DataFrame({
                "Valore Reale": y_test.flatten(),
                "Predizione": y_pred.flatten()
            })
        else:
            columns_real = [f"Valore Reale {i+1}" for i in range(num_outputs)]
            columns_pred = [f"Predizione {i+1}" for i in range(num_outputs)]
            comparison_df = pd.DataFrame(np.hstack([y_test, y_pred]), columns=columns_real + columns_pred)
        
        comparison_path = os.path.join(self.folder, "predictions_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Confronto predizioni salvato in {comparison_path}")
        return comparison_df
    
    def plot_training_metrics(self):
        if self.history is None:
            print("Nessuna history trovata. Assicurati di aver eseguito il training con set di validazione.")
            return
        
        metrics = list(self.history['validation_0'].keys())
        num_epochs = len(next(iter(self.history['validation_0'].values())))
        epochs = range(num_epochs)

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # Training set
            plt.plot(epochs, self.history['validation_0'][metric], label='Train', marker='o')
            
            # Validation set (se presente)
            if 'validation_1' in self.history:
                plt.plot(epochs, self.history['validation_1'][metric], label='Validation', marker='s')

            plt.title(f"Andamento della metrica '{metric.upper()}'")
            plt.xlabel("Numero di alberi (boosting rounds)")
            plt.ylabel(metric.upper())
            plt.legend()
            plt.grid(True)
            
            # Mostra il grafico
            plt.tight_layout()
            plt.show()

            # Salva il grafico
            filename = f"{metric}_trend.png"
            plot_path = os.path.join(self.folder, filename)
            plt.savefig(plot_path)
            print(f"Grafico '{metric}' salvato in: {plot_path}")
            plt.close()