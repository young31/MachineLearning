import numpy as np

class Game:
    def __init__(self, state, FIRST=1):
        self.state = state
        self.empty = self.make_empty(state)
        self.first_player = FIRST
        
    def make_empty(self, state):
        emp = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    emp.append(3*i + j)
        
        return emp
    

    def is_lose(self):
        for i in range(3):
            if self.state[i][0] == self.state[i][1] == self.state[i][2] != 0:
                return True
            elif self.state[0][i] == self.state[1][i] == self.state[2][i] != 0:
                return True
        if self.state[0][0] == self.state[1][1] == self.state[2][2] != 0:
            return True
        if self.state[0][2] == self.state[1][1] == self.state[2][0] != 0:
            return True
        return 0
    

    def is_draw(self):
        a = self.next_opp()
        if self.is_lose():
            return 0
        if np.all(self.state):
            return 1
        else:
            return 0
        

    def is_done(self):
        if self.is_lose() or self.is_draw():
            return 1
        else:
            return 0
        
        
    def update(self, target):
        state = self.state.copy()
        x, y = target//3, target%3
        a = self.next_opp()
        state[x][y] = a
        return Game(state)
    
    
    def next_opp(self):
        a = b = 0
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                if self.state[i][j] == self.first_player:
                    a += 1
                elif self.state[i][j] != 0:
                    b += 1
                    
        if a == b:
            return self.first_player
        else:
            return 2 + min(0, 1-self.first_player)