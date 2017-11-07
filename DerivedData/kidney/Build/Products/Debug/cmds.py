import os
from itertools import product

er = [5]
dr = [.1]
tl = [50]
thor = [2, 10]
rhor = [2, 10]
scl = [.01, .1, 2, 10, 100]
nsim = [3, 20]
nrol = [1, 10]

for args in product(er, dr, tl, scl, thor, rhor, nsim, nrol):
    os.system("time ./kidney {} {} {} {} {} {} {} {} | grep -v 'Aca*'".format(*args))


