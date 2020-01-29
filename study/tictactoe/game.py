import numpy as np

class Game:
    def __init__(self, state):
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
    

    def is_lose(self, a):
        opp = 2 - (a-1)
        
        for i in range(3):
            if self.state[i][0] == self.state[i][1] == self.state[i][2] == opp:
                return 1
            elif self.state[0][i] == self.state[1][i] == self.state[2][i] == opp:
                return 1
        if self.state[0][0] == self.state[1][1] == self.state[2][2] == opp:
            return 1
        return 0
    

    def is_win(self, a):       
        for i in range(3):
            if self.state[i][0] == self.state[i][1] == self.state[i][2] == a:
                return 1
            elif self.state[0][i] == self.state[1][i] == self.state[2][i] == a:
                return 1
        if self.state[0][0] == self.state[1][1] == self.state[2][2] == a:
            return 1
        if self.state[0][2] == self.state[1][1] == self.state[2][0] == a:
            return 1
        return 0
    

    def is_draw(self, a):
        if self.is_win(a):
            return 0
        if np.all(self.state):
            return 1
        else:
            return 0
        

    def is_done(self):
        if self.is_win(1) or self.is_win(2) or self.is_draw(a):
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


    def __str__(self):
        k = np.where(self.state==0, '-', np.where(a==1, 'O', 'X'))
        return k

class MCS:
    def __init__(self, status, n):
        self.status = status
        self.n = n
    

    def playout(self, game):
        if game.is_lose(self.status):
            return -1
        
        if game.is_draw(self.status):
            return 0
        
        if game.is_win(self.status):
            return 1
        
        return self.playout(game.update(np.random.choice(game.empty)))
    
    
    def action(self, game):
        values = [0] * len(game.empty)

        for i, a in enumerate(game.empty):
            for _ in range(self.n):
                g = game.update(a)
                values[i] += self.playout(g)
                
        return game.empty[np.argmax(values)]
    
def play(game, m):
    global score
    while 1:
        if FIRST == 1:
            print('choose among available, \n', game.empty)
            a1 = int(input())
            while a1 not in game.empty:
                a1 = int(input(f'$a1 is not available \n choose again \n'))
        else:
            a1 = m.action(game)

        game = game.update(a1)
        if game.is_win(m.status):
            score[m.status-1] += 1
            print('선수 승')
            return 
        elif game.is_draw(m.status):
            score[2] += 1
            print('무승부')
            return 
        print(game.state)

        if FIRST == 2:
            print('choose among available, \n', game.empty)
            a2 = int(input())
            while a2 not in game.empty:
                a2 = int(input(f'$a2 is not available \n choose again \n'))
        else:
            a2 = m.action(game)

        game = game.update(a2)
        if game.is_win(m.status):
            score[m.status-1] += 1
            print('후수 승')
            return 
        elif game.is_draw(m.status):
            score[2] += 1
            print('무승부')
            return 
        print(game.state)


if __name__ == "__main__":
    init_state = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    score = [0, 0, 0]
    FIRST = int(input('선수: 1, 후수: 2 \n'))
    g = Game(init_state)
    com = 2 + min(0, 1-FIRST)
    m = MCS(com, 100)
    seed = 0
    while 1:
        seed += 1
        np.random.seed(seed)
        play(g, m)
        print(score)
        print('################')