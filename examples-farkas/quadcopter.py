import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint, HyperRectangle
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.plotutil import PlotSettings
from hylaa.star import init_hr_to_star
from hylaa.timerutil import Timers
from hylaa.pv_container import PVObject
from controlcore.control_utils import get_input, extend_a_b
from farkas_central.bdd4Ce import BDD4CE
from hylaa.ce_smt import CeSmt
from hylaa.ce_milp import CeMilp
from hylaa.simutil import compute_simulation
import matplotlib.pyplot as plt

'''PID, LQR and LQR-PID on a Quadcopter Platform'''
''' Lucas M. Argentim '''


def define_ha(settings, usafe_r=None):
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2", "x3", "x4", "x5", "x6"]
    #
    loc1 = ha.new_mode('loc1')
    a_matrix = np.array([[0, 1, 0, 0, 0, 0], [0, 0, -0.0106, 0.0106, -0.0106, 0.0106], [0, 0, -10, 0, 0, 0],
                         [0, 0, 0, -10, 0, 0], [0, 0, 0, 0, -10, 0], [0, 0, 0, 0, 0, -10]], dtype=float)

    b_matrix = np.array([[0], [0], [1], [-1], [1], [-1]], dtype=float)

    print(a_matrix,  b_matrix)
    R_mult_factor = 0.01

    Q_matrix = np.eye(len(a_matrix[0]), dtype=float)

    u_dim = len(b_matrix[0])
    R_matrix = R_mult_factor * np.eye(u_dim)

    print(a_matrix, b_matrix, Q_matrix, R_matrix)
    k_matrix = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)

    print(k_matrix)
    a_bk_matrix = a_matrix - np.matmul(b_matrix, k_matrix)

    loc1.a_matrix = a_bk_matrix
    loc1.c_vector = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    # print(a_bk_matrix)

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:

        # exp 1
        # usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], -0.4))

        # exp 2
        usafe_set_constraint_list.append(LinearConstraint([0.0, 0.0, -1.0, 0.0, 0.0, 0.0], -0.41))
        # To find the error (reverse - number of counterexamples reduced in equ run)
        # usafe_set_constraint_list.append(LinearConstraint([0.0, 0.0, -1.0, 0.0, 0.0, 0.0], -0.47))

        # exp 3 To find the error (reverse - number of counterexamples increased in equ run)
        # Gets fixed with following values
        #         bigM = 10000.0
        #         lb = -1000.0
        #         ub = 1000.0
        # or epsilon2 = 0.00001 (Earlier it was 0.0005)
        # usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], -0.4))
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    trans = ha.new_transition(loc1, error)
    for constraint in usafe_set_constraint_list:
        trans.condition_list.append(constraint)

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, v]
    rv = [(ha.modes['loc1'], init_r)]

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 0
    plot_settings.ydim = 2

    settings = HylaaSettings(step=0.1, max_time=10.0, disc_dyn=False, plot_settings=plot_settings)
    settings.stop_when_error_reachable = False

    return settings


def run_hylaa(settings, init_r, usafe_r):
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)

    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


if __name__ == '__main__':
    settings = define_settings()

    # exp 1 and 2
    init_r = HyperRectangle([(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)])

    # exp 3
    # init_r = HyperRectangle([(-0.1, 0.1), (-0.1, 0.1), (-0.05, 0.05), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)])

    pv_object = run_hylaa(settings, init_r, None)

    # longest_ce = pv_object.compute_longest_ce()

    # mid-order = +2
    # random: [15, 12, 17, 5, 18, 16, 7, 13, 14, 1, 11, 6, 0, 2, 9, 10, 3, 8, 4]
    bdd_ce_object = BDD4CE(pv_object, equ_run=False, smt_mip='mip')
    bdd_graphs = bdd_ce_object.create_bdd_w_level_merge(level_merge=0, order='mid-order')
    valid_exps, invalid_exps = bdd_graphs[0].generate_expressions()
    print(len(valid_exps), len(invalid_exps))

    Timers.print_stats()

