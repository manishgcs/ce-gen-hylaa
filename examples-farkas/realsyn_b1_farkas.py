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


def define_ha(settings, usafe_r=None):
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2"]
    #
    loc1 = ha.new_mode('loc1')
    # a_matrix = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=float)

    # exp 1 and 2
    # a_matrix = np.array([[0.0, 2.0], [-1.5, 0.0]], dtype=float)

    # exp 3
    a_matrix = np.array([[0.0, 2.0], [-1.5, -1.0]], dtype=float)


    # exp 1
    # b_matrix = np.array([[1], [-1]], dtype=float)

    # exp2
    b_matrix = np.array([[1], [1]], dtype=float)

    print(a_matrix,  b_matrix)
    R_mult_factor = 0.2

    Q_matrix = np.eye(len(a_matrix[0]), dtype=float)

    u_dim = len(b_matrix[0])
    R_matrix = R_mult_factor * np.eye(u_dim)

    print(a_matrix, b_matrix, Q_matrix, R_matrix)
    k_matrix = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)

    print(k_matrix)
    # a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
    a_bk_matrix = a_matrix - np.matmul(b_matrix, k_matrix)

    loc1.a_matrix = a_bk_matrix
    loc1.c_vector = np.array([0.0, 0.0], dtype=float)
    # print(a_bk_matrix)

    # loc1.a_matrix = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=float)
    # loc1.c_vector = np.array([0.0, -9.81], dtype=float)

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:
        # exp 1
        # usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0], -2.1))

        # exp 2 - Significant diff across equivalent and non-equivalent runs for p_intersect reverse
        # usafe_set_constraint_list.append(LinearConstraint([0.0, 1.0], -0.75))

        # exp 3 - Significant diff (10%) across equivalent and non-equivalent runs for p_intersect w/o reverse
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 1.0], -1.0))

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
    plot_settings.ydim = 1

    settings = HylaaSettings(step=0.02, max_time=2.0, disc_dyn=False, plot_settings=plot_settings)
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

    # exp 1
    # init_r = HyperRectangle([(1.0, 1.5), (1.0, 1.5)])

    # exp 2
    # init_r = HyperRectangle([(2.0, 2.5), (2.0, 2.5)])

    # exp 3
    init_r = HyperRectangle([(1.5, 2.5), (1.5, 2.5)])

    pv_object = run_hylaa(settings, init_r, None)

    longest_ce = pv_object.compute_longest_ce()

    ce_smt_object = CeSmt(pv_object)
    ce_smt_object.compute_counterexample()
    # ce_smt_object.compute_counterexample(regex=["01111111110111111111"])
    ce_mip_object = CeMilp(pv_object)
    ce_mip_object.compute_counterexample('Ball')
    bdd_ce_object = BDD4CE(pv_object)
    bdd_ce_object.create_bdd()

    Timers.print_stats()