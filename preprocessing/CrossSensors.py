# CrossSensors.py

import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_rows", None)  
pd.set_option("display.max_columns", None)  


class CrossSensors:
    def __init__(self, microphones):
        self.microphones = microphones

    def __repr__(self):
        return f"CrossSensors(microphones={repr(self.microphones)})"
    
    def to_dataframe(self):
        # merges the dataframe of each microphone in a single datafram
        return pd.concat([mic.to_dataframe() for mic in self.microphones], ignore_index=True)
    
    def plot_sensors(self):
        # plots the sensor positions using x and y coordinates
        df = self.to_dataframe()
        
        # Try to maximize the window if in a local environment
        try:
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()  # Attempt to maximize the window
        except AttributeError:
            # If the above line fails (in environments like Jupyter), we just set a large figure size
            pass
        
        # Set the figure size for larger resolution
        plt.figure(figsize=(20, 12)) 
        
        plt.scatter(df['x'], df['y'], c='blue', label='Microphones', alpha=0.7)
        
        # Annotate each point with its name
        for _, row in df.iterrows():
            plt.text(row['x'], row['y'], row['name'], fontsize=5, ha='right', va='bottom')
        
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Microphone Positions")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()


























