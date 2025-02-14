# Microphone.py

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
        return f"Microphone({self.mic}, {self.name}, {self.x}, {self.y}, {self.dx}, {self.dy}, {self.p}, {self.q})"