import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


class RandomForestModel:
    def __init__(self, folder, n_estimators=100, max_depth=None, random_state=42):

    ### FIELDS ###

         # model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            oob_score=True,
            max_depth=max_depth,
            random_state=random_state
        )

        # base fields
        self.folder = folder
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        # evaluation fields
        self.metrics = None
        self.training_time = None
        
        # methods
        self.create_results_folder(folder)


    ###################
    ### GET METHODS ###

    # restituisce i parametri del modello
    def get_model_params(self):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
        }

    def get_obs_score(self):
        return {
            'oob_score': self.model.oob_score_
        }
    
    # restituisce il training time
    def get_training_time(self):
        return {
            'training_time': self.training_time
        }
    
    # restituisce le metriche del trained model
    def get_metrics_summary(self):
        return {
            'output': self.metrics["Output"][0],
            'mse': self.metrics["MSE"][0],
            'rmse': self.metrics["RMSE"][0],
            'mae': self.metrics["MAE"][0],
            'r2': self.metrics["R2"][0]
        }
    
    # restituisce tutte le informazioni
    def get_full_summary(self):
        summary = {}
        summary.update(self.get_model_params())
        summary.update(self.get_obs_score())
        summary.update(self.get_metrics_summary())
        summary.update(self.get_training_time())
        return summary
    


    #################
    ### TRAINING  ###

    # training method
    def train(self, x_train, y_train):
        # set starting time
        start_time = time.time()

        # fit
        self.model.fit(x_train, y_train)

        # compute training time
        self.training_time = time.time() - start_time

        # observe score
        print("OOB Score:", self.model.oob_score_)

        
    ##########################################
    ### PREDICTIONS AND EVALUATION METHODS ###
    
    # compute predictions
    def predict(self, x_test):
        return self.model.predict(x_test)


    # evaluation method
    def evaluate(self, x_test, y_test):
        # predictions
        y_pred = self.predict(x_test)

        # handling of the two cases, monofactorial or multifactorial output
        num_outputs = y_test.shape[1]

        if num_outputs == 1:
            y_test = y_test.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)

        detailed_results = {}

        if num_outputs == 1:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.metrics = {
                "Output": ["Unico"],
                "MSE": [mse],
                "RMSE": [rmse],
                "MAE": [mae],
                "R2": [r2]
            }
        else:
            detailed_results["Output"] = [f"Output {i+1}" for i in range(num_outputs)]
            detailed_results["MSE"] = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            detailed_results["RMSE"] = [np.sqrt(mse) for mse in detailed_results["MSE"]]
            detailed_results["MAE"] = [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            detailed_results["R2"] = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]

            self.metrics = {
                "Output": ["Media"],
                "MSE": [np.mean(detailed_results["MSE"])],
                "RMSE": [np.mean(detailed_results["RMSE"])],
                "MAE": [np.mean(detailed_results["MAE"])],
                "R2": [np.mean(detailed_results["R2"])]
            }

    # compare predictions with true values
    def compare_predictions(self, X_test, y_test):
        # make predictions
        y_pred = self.predict(X_test)

        # handling of the two cases, monofactorial or multifactorial output
        num_outputs = y_test.shape[1]
        
        if num_outputs == 1:
            comparison_df = pd.DataFrame({
                "Real Values": y_test.flatten(),
                "Predicted Values": y_pred.flatten()
            })
        else:
            columns_real = [f"Valore Reale {i+1}" for i in range(num_outputs)]
            columns_pred = [f"Predizione {i+1}" for i in range(num_outputs)]
            comparison_df = pd.DataFrame(np.hstack([y_test, y_pred]), columns=columns_real + columns_pred)
        
        comparison_path = os.path.join(self.folder, "predictions_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Confronto predizioni salvato in {comparison_path}")
        return comparison_df
    

    ###########################
    ### SAVE AND LOAD MODEL ###

    def save_model(self):
        """Salva il modello addestrato."""
        path = os.path.join(self.folder, "random_forest_model.pkl")
        joblib.dump(self.model, path)
        print(f"Modello salvato in {path}")

    def load_model(self):
        """Carica un modello esistente."""
        path = os.path.join(self.folder, "random_forest_model.pkl")
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Modello caricato da {path}")
        else:
            print("Modello non trovato.")


    #################
    ### UTILITIES ###

    # create folder results
    def create_results_folder(self, folder):
        os.makedirs(folder, exist_ok=True)











