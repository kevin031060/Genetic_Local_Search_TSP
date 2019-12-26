import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import LS_2opt
import time


def dist(dist1, dist2, w1, w2, tour):
    dists = [dist1[x1, x2] for x1, x2 in zip(tour[:-1], tour[1:])]
    obj1 = sum(dists) + dist1[tour[0], tour[-1]]
    dists = [dist2[x1, x2] for x1, x2 in zip(tour[:-1], tour[1:])]
    obj2 = sum(dists) + dist2[tour[0], tour[-1]]
    return w1 * obj1 + w2 * obj2, obj1, obj2

static_size, N = 4, 200
tour_rl = scio.loadmat('data/tour%d_%d.mat' % (static_size, N))['tour'].squeeze(1)
dist1 = scio.loadmat('data/obj1_%d_%d.mat' % (static_size, N))['obj1']
dist2 = scio.loadmat('data/obj2_%d_%d.mat' % (static_size, N))['obj2']
rl = scio.loadmat('data/rl%d_%d.mat' % (static_size, N))['rl']

t1 = time.time()
wlist=np.arange(tour_rl.shape[0])/100
objs_ori = []
objs_ls = []
for i in range(tour_rl.shape[0]):
    print(1 - wlist[i], wlist[i])
    # _, o1, o2 = dist(dist1, dist2, 1 - wlist[i], wlist[i], tour_rl[i])
    # objs_ori.append([o1, o2])
    tour_ls = LS_2opt.ls_2opt(dist1, dist2, 1-wlist[i], wlist[i], tour_rl[i], 300)
    _, o1, o2 = dist(dist1, dist2, 1-wlist[i], wlist[i], tour_ls)
    objs_ls.append([o1, o2])
objs_ori = np.array(objs_ori)
objs_ls = np.array(objs_ls)

print(time.time()-t1)
# plt.scatter(objs_ori[:,0], objs_ori[:,1], c='b')
plt.scatter(objs_ls[:,0], objs_ls[:,1], c='r')
plt.scatter(rl[:,0], rl[:,1], c='y')
plt.show()

import numpy as np
# def DPX_crossover(parent1, parent2):
#     # '6 0 4 5 8 9 7 3 2 1'
#     p1 = ' '.join([str(x) for x in parent1])
#     p2 = ' ' + ' '.join([str(x) for x in parent2]) + ' '
#     common_arcs = ''
#     idx_space = -1
#     # start of the sub arc
#     idx_start = 0
#     while True:
#         # p1[0:idx + 1]: '6' --->  '6 0' ---> ... until the sub arc is not found in p2
#         idx_end = p1.find(' ', idx_space + 1)
#
#         sub_arc = p1[idx_start:idx_end] if idx_end != -1 else p1[idx_start:]
#
#         # find sub-arc in p2. [::-1] reversed arc also counts
#         exist = p2.find(' '+sub_arc.strip()+' ') != -1 or p2.find(' '+sub_arc[::-1].strip()+' ') != -1
#         if exist:
#             # idx_space: -1--> 1.
#             idx_space = p1.find(' ', idx_space + 1)
#             sub_arc_pre = sub_arc
#             if idx_space == -1:
#                 break
#         else:
#             common_arcs = common_arcs + sub_arc_pre + ' ,'
#             # the node after the space
#             idx_start = idx_space + 1
#     common_arcs = common_arcs + sub_arc_pre + ' ,'
#     tmp = ' '.join(map(str.strip, np.random.permutation(common_arcs.split(',')).tolist()))
#     offspring = [int(x) for x in tmp.split(' ') if x]
#
#     return offspring
#
# if __name__ == '__main__':
#     p1=np.array([11, 6, 0, 4, 5, 8, 9, 7, 3, 2, 10, 1])
#     p2=np.array([11, 10, 7, 2, 3, 6, 0, 4, 5, 9, 8, 1])
#     # p = [np.random.permutation(np.arange(20)) for _ in range(10)]
#     # p1 = p[np.random.randint(10)]
#     # p2 = p[np.random.randint(10)]
#     t1= str(p1)
#     t2= str(p2)
#     a=DPX_crossover(p1,p2)
#     print('o')