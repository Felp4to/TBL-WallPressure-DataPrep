import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LSTMModel:
    def __init__(self, input_shape, lstm_units=50, output_units=1, dropout_rate=0.2, learning_rate=0.001, return_sequences=False):
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
            Dense(output_units)  # Nessuna attivazione per forecasting
        ])

        # Compilazione con MSE e RMSE
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss="mse",
                           metrics=["mae", self.root_mean_squared_error])
        
        self.history = None  # Variabile per salvare la history del training
    
    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
    def summary(self):
        """Mostra il sommario del modello."""
        self.model.summary()
    
    def train(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None):
        """Addestra il modello e salva la history."""
        self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
        return self.history
    
    def predict(self, x_test):
        """Genera previsioni."""
        return self.model.predict(x_test)
    
    def evaluate(self, x_test, y_test):
        """Valuta il modello sul test set e calcola metriche di errore."""
        y_pred = self.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R^2 Score: {r2:.4f}")
        
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
    
    def save_training_history(self, path="training_history.npy"):
        """Salva la history del training su file."""
        if self.history is not None:
            np.save(path, self.history.history)
            print(f"Training history salvata in {path}")
        else:
            print("Nessuna history trovata. Esegui il training prima di salvare.")
    
    def plot_predictions(self, y_test, y_pred):
        """Confronta le predizioni con i valori di test."""
        plt.figure(figsize=(10, 5))
        plt.plot(y_test, label='Valori Reali', color='blue')
        plt.plot(y_pred, label='Predizioni', color='red', linestyle='dashed')
        plt.legend()
        plt.title('Confronto tra Predizioni e Valori Reali')
        plt.show()
    
    def save_model(self, path="lstm_model.h5"):
        """Salva il modello."""
        self.model.save(path)
    
    def load_model(self, path="lstm_model.h5"):
        """Carica un modello esistente."""
        self.model = tf.keras.models.load_model(path, custom_objects={"root_mean_squared_error": self.root_mean_squared_error})
