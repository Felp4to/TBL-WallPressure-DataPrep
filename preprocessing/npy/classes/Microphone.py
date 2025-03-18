# Microphone.py

import pandas as pd

pd.set_option("display.max_rows", None)  
pd.set_option("display.max_columns", None)  


class Microphone:
    def __init__(self, id, name, x, y, dx, dy, p, q):
        self.id = id
        self.name = name
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.p = p
        self.q = q

    def __repr__(self):
        return f"Microphone({self.id}, {self.name}, {self.x}, {self.y}, {self.dx}, {self.dy}, {self.p}, {self.q})"
    
    def to_dataframe(self):
        # return dataframe with only one row and one column for each attribute
        return pd.DataFrame([ {"id": self.id,
                            "name": self.name,
                            "x": self.x,
                            "y": self.y,
                            "dx": self.dx,
                            "dy": self.dy,
                            "p": self.p,
                            "q": self.q }])