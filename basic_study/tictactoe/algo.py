import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import keras.backend as K

class Random:
    def action(self, game):
        return np.random.choice(game.empty)


class MCS:
    def __init__(self, n=100):
        self.n = n
    
    def playout(self, game):
        if game.is_lose():
            return -1
        
        if game.is_draw():
            return 0

        return -self.playout(game.update(np.random.choice(game.empty)))
    
    
    def action(self, game):
        values = [0] * len(game.empty)

        for i, a in enumerate(game.empty):
            for _ in range(self.n):
                g = game.update(a)
                values[i] -= self.playout(g)

        return game.empty[np.argmax(values)]


class Alpha:
    def value(self, game, alpha, beta):
        if game.is_lose():
            return -1
        
        if game.is_draw():
            return 0

        best_score = -float('inf')
        score = 0
        for a in game.empty:
            score -= self.value(game.update(a), -beta, -alpha)
            
            if score > alpha:
                alpha = score
            if alpha >= beta:
                return alpha
                
        return alpha
        
    def action(self, game):
        best_action = game.empty[0]
        alpha = -float('inf')
        
        for a in game.empty:
            score = -self.value(game.update(a), -float('inf'), -alpha)
            if score > alpha:
                best_action = a
                alpha = score
        return best_action


class CNN: # beta version
    def __init__(self):
        self.model = models.load_model('./CNN.h5')

    def action(self, game):
        res = self.predict(game)
        a = np.argmax(res)
        a = game.empty[a]

        return a

    
    def make_state(self, game):
        status = game.next_opp()
        opp = 3 - status
        a = game.state
        a1 = np.where(a==status, 1, 0)
        a2 = np.where(a==opp, 1, 0)
        res = np.array([a1, a2])
        res = res.reshape(2, 3, 3).transpose(1, 2, 0).reshape(1, 3, 3, 2)
        
        return res
    
    def predict(self, game):
        state = self.make_state(game)
        res = self.model.predict(state)[0]
        res = res[game.empty]
        
        return res 

    