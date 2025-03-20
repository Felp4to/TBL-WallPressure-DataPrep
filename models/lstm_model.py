import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore



class LSTMModel:
    def __init__(self, input_shape, folder, lstm_units=50, output_units=1, dropout_rate=0.2, learning_rate=0.001, return_sequences=False, patience=5):
        """
        Costruttore della classe LSTMModel per forecasting.
        :param input_shape: Tuple (timesteps, features) per i dati di input.
        :param lstm_units: Numero di unità LSTM.
        :param output_units: Numero di output (1 per regressione, più per multivariate).
        :param dropout_rate: Percentuale di dropout per la regolarizzazione.
        :param learning_rate: Tasso di apprendimento per l'ottimizzatore Adam.
        :param return_sequences: Se True, l'ultimo layer LSTM mantiene la sequenza.
        """
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units, return_sequences=return_sequences),
            Dropout(dropout_rate),
            Dense(output_units)
        ])

        # Compilazione con MSE e RMSE
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss="mse",
                           metrics=["mae", self.root_mean_squared_error])
        
        self.history = None  

        self.folder = folder #os.path.join("./tests", folder)
        self.create_folder() 
        self.patience = patience

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
    def create_folder(self):
        """Crea la cartella di output se non esiste già."""
        os.makedirs(self.folder, exist_ok=True)
    
    def summary(self):
        """Mostra il sommario del modello."""
        self.model.summary()

    def save_summary(self):
        """Salva il sommario del modello in un file di testo nella cartella specificata."""
        summary_path = os.path.join(self.folder, "model_summary.txt")
        with open(summary_path, "w") as f:
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))
        print(f"Sommario del modello salvato in {summary_path}")
    
    #def train(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None):
        """Addestra il modello e salva la history."""
    #    self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
    #    return self.history
    
    def train(self, x_train, y_train, epochs=100, batch_size=32, validation_data=None):
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True
        )

        checkpoint_path = os.path.join(self.folder, "best_model.keras")
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        )

        start_time = time.time()  # Inizio misurazione tempo di addestramento
        
        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping, model_checkpoint]
        )
        
        training_time = time.time() - start_time  # Tempo totale di addestramento
        
        # Salvataggio tempo di addestramento
        time_path = os.path.join(self.folder, "training_time.txt")
        with open(time_path, "w") as f:
            f.write(f"Tempo totale di addestramento: {training_time:.2f} secondi\n")
        
        print(f"Tempo totale di addestramento salvato in {time_path}")
        
        # Salvataggio della history in un file di testo
        history_path = os.path.join(self.folder, "training_history.txt")
        with open(history_path, "w") as f:
            for key, values in self.history.history.items():
                f.write(f"{key}: {values}\n")
        
        print(f"Training history salvata in {history_path}")
        
        return self.history

    def predict(self, x_test):
        """Genera previsioni."""
        return self.model.predict(x_test)
    
    def evaluate(self, x_test, y_test):
        """Valuta il modello sul test set e salva i risultati delle metriche."""
        y_pred = self.predict(x_test)

        # Controllo coerenza forma tra y_test e y_pred
        if y_test.shape != y_pred.shape:
            raise ValueError(f"Errore: la forma di y_test {y_test.shape} e y_pred {y_pred.shape} non corrisponde.")

        num_outputs = y_test.shape[1]  # Numero di variabili di output

        detailed_results = {}  # Metriche per ogni output
        mean_results = {}  # Metriche medie su tutti gli output

        if num_outputs == 1:
            # Caso univariato
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
            mean_results = detailed_results  # Per uniformità, la media è uguale ai risultati

        else:
            # Caso multivariato: metriche per ogni output
            detailed_results["Output"] = [f"Output {i+1}" for i in range(num_outputs)]
            detailed_results["MSE"] = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            detailed_results["RMSE"] = [np.sqrt(detailed_results["MSE"][i]) for i in range(num_outputs)]
            detailed_results["MAE"] = [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]
            detailed_results["R2"] = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(num_outputs)]

            # Calcolo delle metriche medie
            mean_results = {
                "Output": ["Media"],
                "MSE": [np.mean(detailed_results["MSE"])],
                "RMSE": [np.sqrt(np.mean(detailed_results["MSE"]))],  # RMSE medio dalla media dei MSE
                "MAE": [np.mean(detailed_results["MAE"])],
                "R2": [np.mean(detailed_results["R2"])]
            }

        # Salvataggio delle metriche dettagliate
        detailed_path = os.path.join(self.folder, "evaluation_metrics_detailed.csv")
        pd.DataFrame(detailed_results).to_csv(detailed_path, index=False)
        
        # Salvataggio delle metriche medie
        mean_path = os.path.join(self.folder, "evaluation_metrics_mean.csv")
        pd.DataFrame(mean_results).to_csv(mean_path, index=False)

        print(f"Metriche dettagliate salvate in: {detailed_path}")
        print(f"Metriche medie salvate in: {mean_path}")

        return {
            "detailed": detailed_results,
            "mean": mean_results
        }
    
    def compare_predictions(self, y_test, y_pred):
        """Confronta i valori reali con le predizioni e li salva in un CSV."""
        # Verifica se l'output è monovariato (1) o multivariato (>1)
        if y_test.shape[1] == 1:
            comparison_df = pd.DataFrame({
                "Valore Reale": y_test.flatten(),
                "Predizione": y_pred.flatten()
            })
        else:
            columns_real = [f"Valore Reale {i+1}" for i in range(y_test.shape[1])]
            columns_pred = [f"Predizione {i+1}" for i in range(y_pred.shape[1])]

            comparison_df = pd.DataFrame(np.hstack([y_test, y_pred]), columns=columns_real + columns_pred)

        comparison_path = os.path.join(self.folder, "predictions_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)

        print(f"Confronto predizioni salvato in {comparison_path}")
        return comparison_df
    
    def save_training_history(self):
        """Salva la history del training su file nella cartella specificata."""
        if self.history is not None:
            history_path = os.path.join(self.folder, "training_history.npy")
            np.save(history_path, self.history.history)
            print(f"Training history salvata in {history_path}")
        else:
            print("Nessuna history trovata. Esegui il training prima di salvare.")
        
    def plot_training_history(self):
        """Visualizza e salva i grafici della Loss (MSE), MAE e RMSE durante il training con confronto Training vs Validation."""
        if self.history is None:
            print("Nessuna history trovata. Esegui il training prima di visualizzare i grafici.")
            return
        
        history = self.history.history

        # LOSS (MSE)
        plt.figure(figsize=(8, 5))
        plt.plot(history["loss"], label="Train Loss", color='blue')
        plt.plot(history.get("val_loss", []), label="Validation Loss", color='red', linestyle="dashed")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title("Loss durante il training")
        plt.legend()
        loss_plot_path = os.path.join(self.folder, "loss_plot.png")
        plt.savefig(loss_plot_path)
        plt.show()
        print(f"Grafico Loss salvato in {loss_plot_path}")

        # MAE
        plt.figure(figsize=(8, 5))
        plt.plot(history["mae"], label="Train MAE", color='blue')
        plt.plot(history.get("val_mae", []), label="Validation MAE", color='red', linestyle="dashed")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Error")
        plt.title("MAE durante il training")
        plt.legend()
        mae_plot_path = os.path.join(self.folder, "mae_plot.png")
        plt.savefig(mae_plot_path)
        plt.show()
        print(f"Grafico MAE salvato in {mae_plot_path}")

        # RMSE
        if "root_mean_squared_error" in history:
            plt.figure(figsize=(8, 5))
            plt.plot(history["root_mean_squared_error"], label="Train RMSE", color='blue')
            plt.plot(history.get("val_root_mean_squared_error", []), label="Validation RMSE", color='red', linestyle="dashed")
            plt.xlabel("Epochs")
            plt.ylabel("RMSE")
            plt.title("RMSE durante il training")
            plt.legend()
            rmse_plot_path = os.path.join(self.folder, "rmse_plot.png")
            plt.savefig(rmse_plot_path)
            plt.show()
            print(f"Grafico RMSE salvato in {rmse_plot_path}")
        else:
            print("RMSE non trovato nella history. Verifica la compilazione del modello.") 
    

    def plot_predictions(self, y_test, y_pred):
        """Genera e salva un grafico per confrontare i valori reali e le predizioni.
        
        Supporta sia il caso univariato (1 output) che il caso multivariato (>1 output).
        - Se l'output è univariato, mostra il grafico a video.
        - In entrambi i casi, salva un file PNG del grafico e un CSV con i dati reali e predetti.
        """
        num_outputs = y_test.shape[1]  # Numero di variabili di output
        save_plot_path = os.path.join(self.folder, "predictions_plot.png")
        save_csv_path = os.path.join(self.folder, "predictions_comparison.csv")

        # Creazione del DataFrame per il CSV
        if num_outputs == 1:
            comparison_df = pd.DataFrame({
                "Valore Reale": y_test.flatten(),
                "Predizione": y_pred.flatten()
            })
        else:
            columns_real = [f"Valore Reale {i+1}" for i in range(num_outputs)]
            columns_pred = [f"Predizione {i+1}" for i in range(num_outputs)]
            comparison_df = pd.DataFrame(np.hstack([y_test, y_pred]), columns=columns_real + columns_pred)

        # Salvataggio del CSV
        comparison_df.to_csv(save_csv_path, index=False)
        print(f"File CSV con i valori reali e predetti salvato in: {save_csv_path}")

        # Generazione del grafico
        if num_outputs == 1:
            # Caso univariato: un solo grafico con valori reali e predetti
            plt.figure(figsize=(10, 5))
            plt.plot(y_test, label="Valore Reale", linestyle='dashed', color='blue')
            plt.plot(y_pred, label="Predizione", linestyle='solid', color='red')
            plt.xlabel("Campioni")
            plt.ylabel("Valore")
            plt.title("Confronto Valori Reali vs Predizioni")
            plt.legend()
            plt.grid(True)
            plt.savefig(save_plot_path)
            plt.show()
            print(f"Grafico delle predizioni salvato in: {save_plot_path}")
        

    def save_model(self):
        """Salva il modello nella cartella specificata."""
        model_path = os.path.join(self.folder, "lstm_model.keras")
        self.model.save(model_path)
        print(f"Modello salvato in {model_path}")
    

    def load_model(self):
        """Carica un modello esistente dalla cartella specificata."""
        model_path = os.path.join(self.folder, "lstm_model.keras")
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path, custom_objects={"root_mean_squared_error": self.root_mean_squared_error})
            print(f"Modello caricato da {model_path}")
        else:
            print(f"Nessun modello trovato in {model_path}.")
