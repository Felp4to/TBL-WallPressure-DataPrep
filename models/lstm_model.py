import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore



class LSTMModel:
    def __init__(self, input_shape, folder, lstm_units=50, output_units=1, dropout_rate=0.2, learning_rate=0.001, return_sequences=False):
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
            LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
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

        self.folder = os.path.join("./tests", folder)
        self.create_folder() 

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
            patience=10,
            restore_best_weights=True
        )

        checkpoint_path = os.path.join(self.folder, "best_model.h5")
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
        """Valuta il modello sul test set e salva i risultati delle metriche in un file."""
        y_pred = self.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

        # Salvataggio delle metriche in un file CSV
        results_path = os.path.join(self.folder, "evaluation_metrics.csv")
        pd.DataFrame([results]).to_csv(results_path, index=False)
        
        print(f"Metriche salvate in {results_path}")
        return results
    
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
    
    def plot_predictions(self, y_test, y_pred, slice=50):
        """Confronta le predizioni con i valori di test e salva il grafico."""
        plt.figure(figsize=(15, 8))
        plt.plot(y_test[:slice], label='Valori Reali', color='blue')
        plt.plot(y_pred[:slice], label='Predizioni', color='red', linestyle='dashed')
        plt.legend()
        plt.title('Confronto tra Predizioni e Valori Reali')

        # Salva il grafico nella cartella specificata
        plot_path = os.path.join(self.folder, "predictions_plot_1.png")
        plt.savefig(plot_path)
        plt.close()  # Chiude la figura per evitare che resti in memoria
        
        print(f"Grafico salvato in {plot_path}")

        """Confronta le predizioni con i valori di test e salva il grafico."""
        plt.figure(figsize=(15, 8))
        plt.plot(y_test[:slice*4], label='Valori Reali', color='blue')
        plt.plot(y_pred[:slice*4], label='Predizioni', color='red', linestyle='dashed')
        plt.legend()
        plt.title('Confronto tra Predizioni e Valori Reali')

        # Salva il grafico nella cartella specificata
        plot_path = os.path.join(self.folder, "predictions_plot_2.png")
        plt.savefig(plot_path)
        plt.close()  # Chiude la figura per evitare che resti in memoria
        
        print(f"Grafico salvato in {plot_path}")

        """Confronta le predizioni con i valori di test e salva il grafico."""
        plt.figure(figsize=(15, 8))
        plt.plot(y_test, label='Valori Reali', color='blue')
        plt.plot(y_pred, label='Predizioni', color='red', linestyle='dashed')
        plt.legend()
        plt.title('Confronto tra Predizioni e Valori Reali')

        # Salva il grafico nella cartella specificata
        plot_path = os.path.join(self.folder, "predictions_plot_tot.png")
        plt.savefig(plot_path)
        plt.close()  # Chiude la figura per evitare che resti in memoria
        
        print(f"Grafico salvato in {plot_path}")


    def save_model(self):
        """Salva il modello nella cartella specificata."""
        model_path = os.path.join(self.folder, "lstm_model.h5")
        self.model.save(model_path)
        print(f"Modello salvato in {model_path}")
    
    def load_model(self):
        """Carica un modello esistente dalla cartella specificata."""
        model_path = os.path.join(self.folder, "lstm_model.h5")
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path, custom_objects={"root_mean_squared_error": self.root_mean_squared_error})
            print(f"Modello caricato da {model_path}")
        else:
            print(f"Nessun modello trovato in {model_path}.")
