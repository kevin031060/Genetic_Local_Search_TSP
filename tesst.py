import numpy as np
def DPX_crossover(parent1, parent2):
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
        exist = p2.find(' '+sub_arc.strip()+' ') != -1 or p2.find(' '+sub_arc[::-1].strip()+' ') != -1
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

if __name__ == '__main__':
    p1=np.array([11, 6, 0, 4, 5, 8, 9, 7, 3, 2, 10, 1])
    p2=np.array([11, 10, 7, 2, 3, 6, 0, 4, 5, 9, 8, 1])
    # p = [np.random.permutation(np.arange(20)) for _ in range(10)]
    # p1 = p[np.random.randint(10)]
    # p2 = p[np.random.randint(10)]
    t1= str(p1)
    t2= str(p2)
    a=DPX_crossover(p1,p2)
    print('o')