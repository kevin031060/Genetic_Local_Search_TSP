import numpy as np
from numba import jit


def dist(dist1, dist2, w1, w2, tour):
    dists = [dist1[x1, x2] for x1, x2 in zip(tour[:-1], tour[1:])]
    obj1 = sum(dists) + dist1[tour[0], tour[-1]]
    dists = [dist2[x1, x2] for x1, x2 in zip(tour[:-1], tour[1:])]
    obj2 = sum(dists) + dist2[tour[0], tour[-1]]
    return w1 * obj1 + w2 * obj2


# path1长度比path2短则返回true
def pathCompare(path1, path2, dist1, dist2, w1, w2):
    if dist(dist1, dist2, w1, w2, path1) <= dist(dist1, dist2, w1, w2, path2):
        return True
    return False


def generateRandomPath(bestPath):
    a, b = np.random.choice(np.arange(len(bestPath)),2)
    if a > b:
        return b, a, bestPath[b:a + 1]
    else:
        return a, b, bestPath[a:b + 1]


def reversePath(path):
    rePath = path.copy()
    rePath[1:-1] = rePath[-2:0:-1]
    return rePath


def ls_2opt(dist1, dist2, w1, w2, bestPath, MAXCOUNT):
    count = 0
    while count < MAXCOUNT:

        start, end, path = generateRandomPath(bestPath)
        # print(path)
        rePath = reversePath(path)
        # print(rePath)
        if pathCompare(path, rePath, dist1, dist2, w1, w2):
            count += 1
            continue
        else:
            count = 0
            bestPath[start:end + 1] = rePath
    return bestPath