import numpy as np


def create_polytope_4m_intervals(intervals, n_dims):
    n_constraints = n_dims * 2
    id_matrix = np.identity(n_dims, dtype=float)
    con_matrix = []
    rhs = np.zeros(n_constraints, dtype=float)
    for idx in range(len(id_matrix)):
        interval = intervals[idx]
        row = id_matrix[idx]
        con_matrix.append(-1 * row)
        con_matrix.append(row)
        rhs[idx * n_dims] = -1 * interval[0]
        rhs[idx * n_dims + 1] = interval[1]

    con_matrix = np.array(con_matrix)
    # print(con_matrix, rhs)
    poly = Polytope(n_constraints, con_matrix, rhs, intervals=intervals)

    return poly


class Polytope:
    def __init__(self, n_constraints, con_matrix, rhs, intervals=None):
        self.n_constraints = n_constraints
        self.con_matrix = con_matrix
        self.rhs = rhs
        self.intervals = intervals

    def polytopePrint(self):
        print("Print polytope with " + str(self.n_constraints) + " constraints...")
        print(self.con_matrix)
        print(self.rhs)

    def intersect_w_q(self, q):
        if self.n_constraints == 0:
            i_polytope = Polytope(q.n_constraints, q.con_matrix, q.rhs)
        else:
            i_polytope = Polytope(self.n_constraints, self.con_matrix, self.rhs)
            i_polytope.n_constraints = i_polytope.n_constraints + q.n_constraints
            i_polytope.con_matrix = np.vstack((i_polytope.con_matrix, q.con_matrix))
            i_polytope.rhs = np.append(i_polytope.rhs, q.rhs, axis=0)
        return i_polytope


