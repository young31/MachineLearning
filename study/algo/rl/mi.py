import sys

sys.stdin = open('mi.txt')

mat = []

for i in range(100):
    temp = list(map(int, input()))
    mat.append(temp)

for i in range(10):
    print(mat[i])

import pickle

f = open('test.bin', 'wb')

pickle.dump(mat, f)