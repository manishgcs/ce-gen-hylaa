
class FarkasObject:
    def __init__(self, P1, P2, Q_set, n_vars):
        self.polytope1 = P1
        self.polytope2 = P2
        self.q_set = Q_set
        self.n_state_vars = n_vars