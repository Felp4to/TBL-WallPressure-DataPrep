import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class CNNModel:
    def __init__(self, input_shape, folder, num_filters=64, kernel_size=3, output_units=1,
                 dropout_rate=0.2, learning_rate=0.001, patience=10):
        
        ### FIELDS ###

        # model
        self.model = Sequential([
            Input(shape=input_shape),
            Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'),
            Dropout(dropout_rate),
            Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'),
            Dropout(dropout_rate),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(output_units)
        ])

        # compile# compile with metrics mse and rmse
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss="mse",
                           metrics=["mae", self.root_mean_squared_error])

        # base fields
        self.input_shape = input_shape
        self.folder = folder
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.output_units = output_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.patience = patience
        self.max_epochs = None

        # evaluation fields
        self.history = None
        self.metrics = None
        self.training_time = None
        self.epochs = None

        # methods
        self.create_results_folder(folder) 
        

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    

    ###################
    ### GET METHODS ###

    def get_model_params(self):
        return {
            'input_shape': self.input_shape,
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'output_units': self.output_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'max_epochs': self.max_epochs
        }
    
    # restituisce le metriche del trained model
    def get_metrics_summary(self):
        return {
            'output': self.metrics["Output"][0],
            'mse': self.metrics["MSE"][0],
            'rmse': self.metrics["RMSE"][0],
            'mae': self.metrics["MAE"][0],
            'r2': self.metrics["R2"][0],
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
    
    # return history
    def get_history(self):
        return self.history
    
    # return model summary
    def get_model_summary(self):
        return self.model.summary()
    
    # restituisce tutte le informazioni
    def get_full_summary(self):
        summary = {}
        summary.update(self.get_model_params())
        summary.update(self.get_metrics_summary())
        summary.update(self.get_training_time())
        summary.update(self.get_epochs())
        # Convert history to a dictionary before updating
        history_dict = self.get_history().history 
        summary.update(history_dict) # Update summary with history_dict
        return summary



    #################
    ### TRAINING  ###

    def train(self, x_train, y_train, epochs=100, batch_size=32, validation_data=None):

        # set max epochs and batch size
        self.epochs = epochs
        self.batch_size = batch_size

        # define early stopping
        early_stopping = EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)

        # defien checkpoint system
        checkpoint_path = os.path.join(self.folder, "best_model.keras")
        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_loss",
                                           save_best_only=True, mode="min", verbose=1)

        # set the train start time
        start_time = time.time()

        # fit
        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping, model_checkpoint]
        )

        # compute training time
        self.training_time = time.time() - start_time

        # trainign epochs
        self.epochs = len(self.history.epoch)

    
    ##########################################
    ### PREDICTIONS AND EVALUATION METHODS ###

    # compute predictions
    def predict(self, x_test):
        return self.model.predict(x_test)

    # compute metrics 
    def evaluate(self, x_test, y_test):
        # compute predictions
        y_pred = self.predict(x_test)

        # check shapes
        if y_test.shape != y_pred.shape:
            raise ValueError(f"Errore: la forma di y_test {y_test.shape} e y_pred {y_pred.shape} non corrisponde.")

        # handling of the two cases, monofactorial or multifactorial output
        num_outputs = y_test.shape[1]

        detailed_results = {}  # Metriche per ogni output
        mean_results = {}  # Metriche medie su tutti gli output

        if num_outputs == 1:
            # univariate case
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
            # multivariate case
            detailed_results["Output"] = [f"Output {i+1}" for i in range(num_outputs)]
            detailed_results["MSE"] = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            detailed_results["RMSE"] = [np.sqrt(detailed_results["MSE"][i]) for i in range(num_outputs)]
            detailed_results["MAE"] = [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            detailed_results["R2"] = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]

            # compute average
            self.metrics = {
                "Output": ["Media"],
                "MSE": [np.mean(detailed_results["MSE"])],
                "RMSE": [np.sqrt(np.mean(detailed_results["MSE"]))],
                "MAE": [np.mean(detailed_results["MAE"])],
                "R2": [np.mean(detailed_results["R2"])]
            }
    
    # compare predictions
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
            columns_real = [f"Real Values {i+1}" for i in range(y_test.shape[1])]
            columns_pred = [f"Predicted Values {i+1}" for i in range(y_pred.shape[1])]
            comparison_df = pd.DataFrame(np.hstack([y_test, y_pred]), columns=columns_real + columns_pred)

        comparison_path = os.path.join(self.folder, "predictions_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Confronto predizioni salvato in {comparison_path}")
        return comparison_df


    ###########################
    ### SAVE AND LOAD MODEL ###

    def save_model(self):
        model_path = os.path.join(self.folder, "cnn_model.keras")
        self.model.save(model_path)
        print(f"Modello salvato in {model_path}")

    def load_model(self):
        model_path = os.path.join(self.folder, "cnn_model.keras")
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path, custom_objects={"root_mean_squared_error": self.root_mean_squared_error})
            print(f"Modello caricato da {model_path}")
        else:
            print(f"Nessun modello trovato in {model_path}.")


   #################
    ### UTILITIES ###

    # create folder results
    def create_results_folder(self, folder):
        os.makedirs(folder, exist_ok=True)
          