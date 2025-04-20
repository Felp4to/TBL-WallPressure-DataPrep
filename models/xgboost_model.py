import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump, load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



class XGBoostModel:
    def __init__(self, folder, n_estimators=1000, learning_rate=0.1, max_depth=3, 
                subsample=0.6, colsample_bytree=0.6, patience=15, output_units=1, reg_alpha=1.0, reg_lambda=1.0,
                objective='reg:squarederror', eval_metrics=['rmse', 'mae'], tree_method='auto'):
    
    ### FIELDS ###

        # model
        self.model = xgb.XGBRegressor(  
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective,
            eval_metric=eval_metrics,
            tree_method=tree_method 
        )
        
        # base fields
        self.folder = folder
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.patience = patience
        self.output_units = output_units
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.eval_metrics = eval_metrics
        self.tree_method = tree_method

        # evaluation fields
        self.evals_result = None
        self.metrics = None
        self.training_time = None
        self.epochs = None

        # methods
        self.create_results_folder(folder)


    ###################
    ### GET METHODS ###

    # restituisce i parametri del modello
    def get_model_params(self):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'patience': self.patience,
            'output_units': self.output_units,
            'objective': self.objective,
            'eval_metrics': self.eval_metrics,
            'tree_method': self.tree_method,
        }
    
    # restituisce il risultati del trained model
    def get_evals_result(self):
        return {
            'evals_result': self.evals_result
        }
    
    # restituisce il training time
    def get_training_time(self):
        return {
            'training_time': self.training_time
        }
    
    # restituisce il numero di epoche
    def get_epochs(self):
        return {
            'epochs' : self.epochs
        }
    
    # restituisce le metriche del trained model
    def get_metrics_summary(self):
        return {
            'output': self.metrics["Output"][0],
            'mse': self.metrics["MSE"][0],
            'rmse': self.metrics["RMSE"][0],
            'mae': self.metrics["MAE"][0],
            'r2': self.metrics["R2"][0],
            'mape (%)': self.metrics["MAPE (%)"][0],
            'smape (%)': self.metrics["sMAPE (%)"][0]
        }
    
    # restituisce tutte le informazioni
    def get_full_summary(self):
        summary = {}
        summary.update(self.get_model_params())
        summary.update(self.get_metrics_summary())
        summary.update(self.get_training_time())
        summary.update(self.get_epochs())
        evals_result = self.get_evals_result().get('evals_result', {})
        for eval_set, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                # Prende solo l’ultimo valore per ogni metrica
                summary[f"{eval_set}_{metric_name}"] = values[-1] if isinstance(values, list) else values

        return summary
    

    #################
    ### TRAINING  ###
    
    # training method
    def train(self, x_train, y_train, x_val=None, y_val=None):
        evals_result = {}

        # set start time
        start_time = time.time()

        # convert into DMatrix
        dtrain = xgb.DMatrix(x_train, label=y_train)
        evals = [(dtrain, 'train')]

        if x_val is not None and y_val is not None:
            dval = xgb.DMatrix(x_val, label=y_val)
            evals.append((dval, 'eval'))
        else:
            dval = None
        
        # base parameters
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': self.model.learning_rate,
            'max_depth': self.model.max_depth,
            'subsample': self.model.subsample,
            'colsample_bytree': self.model.colsample_bytree,
            'eval_metric': self.model.eval_metric,
        }

        # training
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.model.n_estimators,
            evals=evals,
            early_stopping_rounds=self.patience if dval else None,
            evals_result=evals_result,
            verbose_eval=True
        )

        # compute training time
        self.training_time = time.time() - start_time

        # save evaluation results
        self.evals_result = evals_result

        # save epochs number
        self.epochs = len(evals_result['train']['rmse'])
    

    ##########################################
    ### PREDICTIONS AND EVALUATION METHODS ###
    
    # compute predictions
    def predict(self, x_test):
        dtest = xgb.DMatrix(x_test)
        y_pred = self.model.predict(dtest)
        return y_pred.reshape(-1, self.output_units)

    # evaluate trained model
    def evaluate(self, x_test, y_test):
        # compute predictions
        y_pred = self.predict(x_test)

        # check shapes
        if y_test.shape != y_pred.shape:
            raise ValueError(f"Errore: la forma di y_test {y_test.shape} e y_pred {y_pred.shape} non corrisponde.")

        # handling of the two cases, monofactorial or multifactorial output
        num_outputs = y_test.shape[1]

        if num_outputs == 1:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = self.mean_absolute_percentage_error(y_test, y_pred)
            smape = self.symmetric_mean_absolute_percentage_error(y_test, y_pred)

            self.metrics = {
                "Output": ["Unico"],
                "MSE": [mse],
                "RMSE": [rmse],
                "MAE": [mae],
                "R2": [r2],
                "MAPE (%)": [mape],
                "sMAPE (%)": [smape]
            }
        else:
            detailed_results = {}
            detailed_results["Output"] = [f"Output {i+1}" for i in range(num_outputs)]
            detailed_results["MSE"] = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            detailed_results["RMSE"] = [np.sqrt(detailed_results["MSE"][i]) for i in range(num_outputs)]
            detailed_results["MAE"] = [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            detailed_results["R2"] = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            detailed_results["MAPE (%)"] = [self.mean_absolute_percentage_error(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            detailed_results["sMAPE (%)"] = [self.symmetric_mean_absolute_percentage_error(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            
            self.metrics = {
                "Output": ["Media"],
                "MSE": [np.mean(detailed_results["MSE"])],
                "RMSE": [np.sqrt(np.mean(detailed_results["MSE"]))],
                "MAE": [np.mean(detailed_results["MAE"])],
                "R2": [np.mean(detailed_results["R2"])],
                "MAPE (%)": [np.mean(detailed_results["MAPE (%)"])],
                "sMAPE (%)": [np.mean(detailed_results["sMAPE (%)"])]
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
            columns_real = [f"Real Values {i+1}" for i in range(num_outputs)]
            columns_pred = [f"Predicted Values {i+1}" for i in range(num_outputs)]
            comparison_df = pd.DataFrame(np.hstack([y_test, y_pred]), columns=columns_real + columns_pred)
        
        comparison_path = os.path.join(self.folder, "predictions_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Confronto predizioni salvato in {comparison_path}")
        return comparison_df
    

    ###########################
    ### SAVE AND LOAD MODEL ###

    # save model
    def save_model(self):
        model_path = os.path.join(self.folder, "xgboost_model.joblib")
        dump(self.model, model_path)
        print(f"Modello salvato in {model_path}")
    
    # load model
    def load_model(self):
        model_path = os.path.join(self.folder, "xgboost_model.joblib")
        if os.path.exists(model_path):
            self.model = load(model_path)
            print(f"Modello caricato da {model_path}")
        else:
            print(f"Nessun modello trovato in {model_path}.")


    #################
    ### UTILITIES ###

    # create folder results
    def create_results_folder(self, folder):
        os.makedirs(folder, exist_ok=True)


    #########################
    ### METRICS AND ERROR ###


    # metrica MAPE indipendente dalla scala
    def mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # metrica sMAPE indipendente dalla scala
    def symmetric_mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        diff = np.abs(y_pred - y_true) / denominator
        # Evitiamo divisioni per zero
        diff[denominator == 0] = 0
        return np.mean(diff) * 100
            


    

