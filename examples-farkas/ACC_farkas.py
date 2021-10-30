'''
Adaptive cruise control, with dynamics from paper by Tiwari
'''

import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, HyperRectangle, LinearConstraint
from hylaa.star import init_hr_to_star
from hylaa.engine import HylaaSettings, HylaaEngine
from hylaa.containers import PlotSettings
from hylaa.pv_container import PVObject
from hylaa.timerutil import Timers
from hylaa.simutil import compute_simulation
from farkas_central.bdd4Ce import BDD4CE
from hylaa.ce_smt import CeSmt
from hylaa.ce_milp import CeMilp


def define_ha(settings, usafe_r):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    # k = -0.0025  # Unsafe
    # k = -1.5  # Safe
    ha = LinearHybridAutomaton()
    ha.variables = ["s", "v", "vf", "a", "t"]

    loc1 = ha.new_mode('loc1')

    # exp 1
    # loc1.a_matrix = np.array([[0, -1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, -3, 2, -2, 0], [0, 0, 0, 0, 0]])

    # exp 2
    # loc1.a_matrix = np.array([[0, -1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [2, -5, 3, -4, 0], [0, 0, 0, 0, 0]])

    # exp 3
    # loc1.a_matrix = np.array([[0, -1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, -4, 3, -3, 0], [0, 0, 0, 0, 0]])

    # exp 4
    # loc1.a_matrix = np.array([[0, -1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, -4, 3, -1.2, 0], [0, 0, 0, 0, 0]])

    # exp 5
    # loc1.a_matrix = np.array([[0, -1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, -3, 2, -2, 0], [0, 0, 0, 0, 0]])

    # exp 6
    # loc1.a_matrix = np.array([[0, -1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, -3, 2, -3.2, 0], [0, 0, 0, 0, 0]])

    # exp 7
    loc1.a_matrix = np.array([[0, -1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, -3, 2, -3.2, 0], [0, 0, 0, 0, 0]])

    # Not stable
    loc1.c_vector = np.array([0, 0, 0, -10, 1], dtype=float)

    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(loc1, error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        # exp 1
        # usafe_set_constraint_list.append(LinearConstraint([0, 1, 0, 0, 0], 12))

        # exp 2
        # usafe_set_constraint_list.append(LinearConstraint([1, 0, 0, 0, 0], 4))

        # exp 3
        # usafe_set_constraint_list.append(LinearConstraint([1, 0, 0, 0, 0], 4.2))

        # exp 4
        # usafe_set_constraint_list.append(LinearConstraint([1, 0, 0, 0, 0], 4.5))

        # exp 5
        # usafe_set_constraint_list.append(LinearConstraint([1, 0, 0, 0, 0], 4.7))

        # exp 6
        # usafe_set_constraint_list.append(LinearConstraint([0, 0, 0, -1, 0], -2))

        # exp 7
        # check for -1.8 as well - similar behavior
        usafe_set_constraint_list.append(LinearConstraint([0, 0, 0, -1, 0], -1.85))
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    for constraint in usafe_set_constraint_list:
        trans.condition_list.append(constraint)

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, y]
    rv = []
    rv.append((ha.modes['loc1'], init_r))

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 1
    plot_settings.ydim = 3

    s = HylaaSettings(step=0.1, max_time=10.0, plot_settings=plot_settings, disc_dyn=False)
    s.stop_when_error_reachable = False
    
    return s


def run_hylaa(settings, init_r, usafe_r):

    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)
    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


if __name__ == '__main__':
    settings = define_settings()
    # init_r = HyperRectangle([(-9, -2), (18, 19), (20, 20), (-1, 1), (0, 0)])
    init_r = HyperRectangle([(2, 5), (18, 22), (20, 20), (-1, 1), (0, 0)])

    pv_object = run_hylaa(settings, init_r, None)
    # longest_ce = pv_object.compute_longest_ce()
    # ce_smt_object = CeSmt(pv_object)
    # ce_smt_object.compute_counterexample()
    # ce_smt_object.compute_counterexample(regex=["01111111110111111111"])
    # ce_mip_object = CeMilp(pv_object)
    # ce_mip_object.compute_counterexample(benchmark='Oscillator')

    # mid-order = +2
    # random: [16, 14, 17, 15, 11, 7, 10, 1, 12, 6, 0, 2, 9, 5, 13, 3, 8, 4]
    bdd_ce_object = BDD4CE(pv_object, equ_run=False, smt_mip='mip')
    bdd_graphs = bdd_ce_object.create_bdd_w_level_merge(level_merge=0, order='mid-order')
    valid_exps, invalid_exps = bdd_graphs[0].generate_expressions()
    print(len(valid_exps), len(invalid_exps))
    Timers.print_stats()

    # psList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # psSet = set()
    # for element in psList:
    #    e_set = set([element])
    #    print ("Final Set: {}".format(e_set))
    #    psSet.add(tuple(e_set))
    #    if psSet.issubset(tuple(e_set)):
    #        print ("This is a subset")
    #
    #    print ("Final Set: {}".format(psSet))
    # pv_object.list_powerset(psList)
    # print(set(pv_object.powerset(psList)))
    # pv_object.powerset(psList)
