import numpy as np
import scipy.io as scio

class GLS:

    def __init__(self):
        self.N = 100
        self.dist1, self.dist2 = self.init_instance(self.N, 4)
        self.K = 20
        self.S = 40

    def init_instance(self, N, static_size):
        obj1 = scio.loadmat('data/obj1_%d_%d.mat' % (static_size, N))['obj1']
        obj2 = scio.loadmat('data/obj2_%d_%d.mat' % (static_size, N))['obj2']
        return obj1, obj2

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
        PE = PE_old.copy()
        if len(PE) == 0:
            PE.append(tour)
            return PE
        else:
            _, obj1, obj2 = self.get_objectives(tour)
            if_nondominated = True
            
            for i in range(len(PE)):
                _, x_obj1, x_obj2 = self.get_objectives(PE[i])
                # if tour can dominate x, remove x
                if obj1<=x_obj1 and obj2<=x_obj2:
                    PE.remove(PE[0])
                # if tour is dominated by anyone of the element in PE, then not add
                if obj1>x_obj1 and obj2>x_obj2:
                    if_nondominated = False
                    break
            if if_nondominated:
                PE.append(tour)

            return PE

    def sort_CS(self, CS, w1, w2):
        scores = np.array([self.get_objectives(x, w1, w2)[0] for x in CS])
        pos = scores.argsort()
        return np.array(CS)[pos], scores

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



    def run(self):
        import LS_2opt

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
            tour_ls = LS_2opt.ls_2opt(self.dist1, self.dist2, w1, w2, x, 100)
            # add to CS
            CS.append(tour_ls)
            # add to PE if it's non-dominated
            PE = self.update_PE(PE, tour_ls)
        print("Done. Begin evolving")
        for i in range(100):
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
            offspring = self.crossover_DPX(parent1, parent2)
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





if __name__ == '__main__':
    gls = GLS()
    gls.run()
