import numpy as np
import algo
from game import Game

class Custom: # make own algo if you want
    def __init__(self):
        return        
    def action(self, game): # this point
        a = np.random.choice(game.empty)
        return a


def play(game):
    global score
    print(visual(game), '\n')
    while 1:
        print('waiting...')
        if FIRST == 1:
            if playmode == 1:
                print('choose among available, \n', game.empty)
                a1 = int(input())
                while a1 not in game.empty:
                    a1 = int(input(f'$a1 is not available \n choose again \n'))
            else:
                a1 = custom.action(game)
        else:
            a1 = m.action(game)

        game = game.update(a1)
        print(visual(game), '\n')
        if game.is_lose() != 0:
            score[0] += 1
            print('선수 승')
            return 
        elif game.is_draw():
            score[2] += 1
            print('무승부')
            return 
        

        if FIRST == 2:
            if playmode == 1:
                print('choose among available, \n', game.empty)
                a2 = int(input())
                while a2 not in game.empty:
                    a2 = int(input(f'$a2 is not available \n choose again \n'))
            else:
                a2 = custom.action(game)
        else:
            a2 = m.action(game)

        game = game.update(a2)
        print(visual(game), '\n')
        if game.is_lose() != 0:
            score[1] += 1
            print('후수 승')
            return 
        elif game.is_draw():
            score[2] += 1
            print('무승부')
            return 
        


def visual(game):
    v = np.where(game.state==0, '-', np.where(game.state==1, 'O', 'X'))
    return v


if __name__ == "__main__":
    playmode = int(input('self: 1 \nalgo: 2 \n'))
    FIRST = int(input('Select your order \n선수: 1, 후수: 2 \n'))
    init_state = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    score = [0, 0, 0]

    g = Game(init_state)
    com = 2 + min(0, 1-FIRST)
    level = int(input('Select game level \n [1, 2, 3, 4] \n'))


    custom = Custom()
    if level == 1:
        m = algo.Random()
    elif level == 2:
        m = algo.CNN()
    elif level == 3:
        m = algo.MCS()
    else:
        m = algo.Alpha()
    
    seed = 0
    
    while 1:
        seed += 1
        np.random.seed(seed)
        play(g)
        print(score)
        print('################')