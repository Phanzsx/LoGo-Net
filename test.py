import numpy as np

a = (0.1112212112, 0.2)
s = ['1', '2']
c = (a, s)
with open('log.txt', 'w') as f:
    for i in a:
        f.write(str(round(i, 2)) + ' ')
    f.write('\n')