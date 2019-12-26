import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import LS_2opt
import time

# find the elements that in l1 but not in l2
def diff(l1, l2):
    tm_list = [np.where(l1 == x) for x in l2]
    return np.delete(l1, tm_list)

def dist(dist1, dist2, w1, w2, tour):
    dists = [dist1[x1, x2] for x1, x2 in zip(tour[:-1], tour[1:])]
    obj1 = sum(dists) + dist1[tour[0], tour[-1]]
    dists = [dist2[x1, x2] for x1, x2 in zip(tour[:-1], tour[1:])]
    obj2 = sum(dists) + dist2[tour[0], tour[-1]]
    return w1 * obj1 + w2 * obj2, obj1, obj2




class GLS:

    def __init__(self, N, static_size, if_rl_ls=True):
        # number of cities
        self.N = N
        self.static_size = static_size
        self.if_rl_ls = if_rl_ls
        # only for two-objectives now
        self.dist1, self.dist2, self.rl = self.init_instance(self.N, self.static_size)
        self.K = 16
        self.S = 140


    def init_instance(self, N, static_size):
        # obj1 = scio.loadmat('data/obj1_%d_%d.mat' % (static_size, N))['obj1']
        # obj2 = scio.loadmat('data/obj2_%d_%d.mat' % (static_size, N))['obj2']
        # rl = scio.loadmat('data/rl%d_%d.mat' % (static_size, N))['rl']
        if self.if_rl_ls:
            self.tour_rl = scio.loadmat('data/tour%d_%d.mat' % (static_size, N))['tour'].squeeze(1)
        dist1 = scio.loadmat('data/obj1_%d_%d.mat' % (static_size, N))['obj1']
        dist2 = scio.loadmat('data/obj2_%d_%d.mat' % (static_size, N))['obj2']
        rl = scio.loadmat('data/rl%d_%d.mat' % (static_size, N))['rl']

        return dist1, dist2, rl

    def init_population(self):
        return [np.random.permutation(np.arange(self.N)) for _ in range(self.S)]

    def random_weight(self):
        w1 = np.random.rand()
        return w1, 1-w1

    def get_objectives(self, tour, w1=1.0, w2=0.0):
        dists = [self.dist1[x1, x2] for x1, x2 in zip(tour[:-1],tour[1:])]
        obj1 = sum(dists) + self.dist1[tour[0], tour[-1]]
        dists = [self.dist2[x1, x2] for x1, x2 in zip(tour[:-1],tour[1:])]
        obj2 = sum(dists) + self.dist2[tour[0], tour[-1]]
        return w1*obj1 + w2*obj2, obj1, obj2

    def update_PE(self, PE_old, tour):
        PE = np.array(PE_old)
        if PE.shape[0] == 0:
            return tour[None,:]

        else:
            objs, obj1, obj2 = self.get_objectives(tour)
            if_nondominated = True
            dominated_list = []
            for i in range(PE.shape[0]):
                objs, x_obj1, x_obj2 = self.get_objectives(PE[i])
                # if tour can dominate x, remove x
                if obj1<x_obj1 and obj2<x_obj2:
                    dominated_list.append(i)
                # if tour is dominated by anyone of the element in PE, then not add
                if obj1>x_obj1 and obj2>x_obj2:
                    if_nondominated = False
                    break
            if len(dominated_list) >0:
                PE = np.delete(PE, dominated_list, axis=0)
            if if_nondominated:
                PE = np.row_stack((PE, tour))

            return PE

    def sort_CS(self, CS, w1, w2):
        scores = np.array([self.get_objectives(x, w1, w2)[0] for x in CS])
        pos = scores.argsort()
        return np.array(CS)[pos], np.sort(scores)

    # Distance-preserve crossover
    def crossover_DPX(self, parent1, parent2):
        # '6 0 4 5 8 9 7 3 2 1'
        p1 = ' '.join([str(x) for x in parent1])
        p2 = ' ' + ' '.join([str(x) for x in parent2]) + ' '
        common_arcs = ''
        idx_space = -1
        # start of the sub arc
        idx_start = 0
        while True:
            # p1[0:idx + 1]: '6' --->  '6 0' ---> ... until the sub arc is not found in p2
            idx_end = p1.find(' ', idx_space + 1)

            sub_arc = p1[idx_start:idx_end] if idx_end != -1 else p1[idx_start:]

            # find sub-arc in p2. [::-1] reversed arc also counts
            exist = p2.find(' ' + sub_arc.strip() + ' ') != -1 or p2.find(' ' + sub_arc[::-1].strip() + ' ') != -1
            if exist:
                # idx_space: -1--> 1.
                idx_space = p1.find(' ', idx_space + 1)
                sub_arc_pre = sub_arc
                if idx_space == -1:
                    break
            else:
                common_arcs = common_arcs + sub_arc_pre + ' ,'
                # the node after the space
                idx_start = idx_space + 1
        common_arcs = common_arcs + sub_arc_pre + ' ,'
        tmp = ' '.join(map(str.strip, np.random.permutation(common_arcs.split(',')).tolist()))
        offspring = [int(x) for x in tmp.split(' ') if x]

        return offspring

    def crossover_classical(self, parent1, parent2):
        pos = np.random.randint(self.N)
        offspring = parent1.copy()
        offspring[pos:] = diff(parent2, parent1[:pos])
        return offspring

    def save(self, ls, rl_ori, rl_ls, ls_count):
        scio.savemat("data/result/rl_ls%d_%d_%d.mat" % (self.static_size, self.N, ls_count), {'rl_ls': rl_ls})
        scio.savemat("data/result/ls%d_%d_%d.mat" % (self.static_size, self.N, ls_count), {'ls': ls})
        scio.savemat("data/result/rl_ori%d_%d_%d.mat" % (self.static_size, self.N, ls_count), {'rl_ori': rl_ori})

    def run(self, ls_count):

        t1=time.time()
        # S initial solutions
        population = self.init_population()
        CS = []
        PE = []
        # Initialize CS and PE
        print("Inital CS and PE")
        for x in population:
            # Uniformly randomly generate a weight vector
            w1, w2 = self.random_weight()
            # Optimize locally by 2-opt local search
            tour_ls = LS_2opt.ls_2opt(self.dist1, self.dist2, w1, w2, x, ls_count)
            # add to CS
            CS.append(tour_ls)
            # add to PE if it's non-dominated
            PE = self.update_PE(PE, tour_ls)


        print("Done. Begin evolving")
        for i in range(10000):
            # Uniformly randomly generate a weight vector
            w1, w2 = self.random_weight()
            # From CS select the K best solutions
            sorted_CS, scores = self.sort_CS(CS, w1, w2)
            TP = sorted_CS[:self.K]
            # Draw at random two solutions from TP
            rand_idx = np.random.choice(np.arange(self.K), 2)
            parent1 = TP[rand_idx[0]]
            parent2 = TP[rand_idx[1]]
            # then generate a new solution
            offspring_ = self.crossover_DPX(parent1, parent2)
            offspring = LS_2opt.ls_2opt(self.dist1, self.dist2, w1, w2, offspring_, ls_count)
            # if offspring is better than the worst solution in TP
            objs, _, _ = self.get_objectives(offspring, w1, w2)
            if objs < scores[self.K-1] and offspring not in TP:
                CS.append(offspring)
            # update PE
            PE = self.update_PE(PE, offspring)
            # if exceeds KS, delete the oldest solution in CS.
            if len(CS) > self.K * self.S:
                CS.remove(CS[0])
            if i % 10 == 0:
                print("Epoch %d:%d"%(i, len(PE)))
        PF_ls = np.array([self.get_objectives(x)[1:] for x in PE])

        if self.if_rl_ls:
            tours = []
            wlist = np.arange(self.tour_rl.shape[0]) / 100
            for i in range(self.tour_rl.shape[0]):
                print(1 - wlist[i], wlist[i])
                # _, o1, o2 = dist(dist1, dist2, 1 - wlist[i], wlist[i], tour_rl[i])
                # objs_ori.append([o1, o2])
                tour_ls = LS_2opt.ls_2opt(self.dist1, self.dist2, 1 - wlist[i], wlist[i], self.tour_rl[i], 300)
                tours.append(tour_ls)
            rl_ls = np.array([self.get_objectives(x)[1:] for x in tours])
        else:
            rl_ls = self.rl

        self.save(PF_ls, self.rl, rl_ls, ls_count)

        print(time.time()-t1)

        plt.scatter(rl_ls[:, 0], rl_ls[:, 1], c='r')
        plt.scatter(PF_ls[:,0], PF_ls[:,1], c='b')
        plt.scatter(self.rl[:, 0], self.rl[:, 1], c='y')
        plt.show()








if __name__ == '__main__':
    # ls_count=[100,200,300]
    # N=[100,200]
    # statics=[3,4]
    # t_list = []
    # for static in statics:
    #     for n in N:
    #         for lc in ls_count:
    #             t1=time.time()
    #             gls = GLS(n, static, if_rl_ls=False)
    #             gls.run(lc)
    #             t = time.time()-t1
    #             t_list.append(t)
    # scio.savemat("time.mat", {'rl_ori': t_list})
    # print(t_list)

    t1 = time.time()
    gls = GLS(100, 4)
    gls.run(10)
    t = time.time() - t1


