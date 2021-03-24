'''
The first two modes of Goran's ball-string example. This is a simple model demonstrating guards and transitions.
'''

import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint, HyperRectangle
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.plotutil import PlotSettings
from hylaa.star import init_hr_to_star
from hylaa.timerutil import Timers
from hylaa.pv_container import PVObject
from farkas_central.bdd4Ce import BDD4CE
from hylaa.ce_smt import CeSmt
from hylaa.ce_milp import CeMilp
from hylaa.simutil import compute_simulation
import matplotlib.pyplot as plt


def define_ha(settings, usafe_r=None):
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x", "v"]

    extension = ha.new_mode('extension')
    extension.a_matrix = np.array([[0.0, 1.0], [-100.0, -4.0]], dtype=float)
    extension.c_vector = np.array([0.0, -9.81], dtype=float)
    extension.inv_list.append(LinearConstraint([1.0, 0.0], 0))  # x <= 0

    freefall = ha.new_mode('freefall')
    freefall.a_matrix = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    freefall.c_vector = np.array([0.0, -9.81], dtype=float)
    freefall.inv_list.append(LinearConstraint([-1.0, 0.0], 0.0))  # 0 <= x
    freefall.inv_list.append(LinearConstraint([1.0, 0.0], 1.0))  # x <= 1

    trans = ha.new_transition(extension, freefall)
    trans.condition_list.append(LinearConstraint([-0.0, -1.0], -0.0))  # v >= 0

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:
        # usafe_set_constraint_list.append(LinearConstraint([0.0, -1.0], -6.5))
        usafe_set_constraint_list.append(LinearConstraint([3.0, -1.0], -7))  # for line with pts (-1, 4) and (0, 7)
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    trans1 = ha.new_transition(extension, error)
    for constraint in usafe_set_constraint_list:
        trans1.condition_list.append(constraint)

    trans2 = ha.new_transition(freefall, error)
    for constraint in usafe_set_constraint_list:
        trans2.condition_list.append(constraint)

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, v]
    rv = [(ha.modes['extension'], init_r)]

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    settings = HylaaSettings(step=0.01, max_time=2.0, disc_dyn=False, plot_settings=plot_settings)
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
    init_r = HyperRectangle([(-1.05, -0.95), (-0.15, 0.15)])
    # usafe_r = HyperRectangle([(-0.2, 0.2), (5, 6)])  # Small
    # usafe_r = HyperRectangle([(-0.5, 0.5), (5, 6.4)])  # To check

    # usafe_r = HyperRectangle([(-0.5, 0.5), (5, 7)])  # Medium
    # usafe_r = HyperRectangle([(-0.5, 0.5), (5, 6.4)])  # Medium (ACC)
    # usafe_r = HyperRectangle([(-0.8, 0.8), (3, 7)])  # Large

    pv_object = run_hylaa(settings, init_r, None)

    longest_ce = pv_object.compute_longest_ce()

    ce_smt_object = CeSmt(pv_object)
    ce_smt_object.compute_counterexample()
    # ce_smt_object.compute_counterexample(regex=["01111111110111111111"])
    ce_mip_object = CeMilp(pv_object)
    ce_mip_object.compute_counterexample('Ball')
    # bdd_ce_object = BDD4CE(pv_object)
    # bdd_ce_object.create_bdd()

    Timers.print_stats()