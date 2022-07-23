'''
Damped oscillator model, with dynamics:

x' == -0.1 * x + y
y' == -x - 0.1 * y
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
import sys
from hylaa.ce_smt import CeSmt
from hylaa.ce_milp import CeMilp
import matplotlib.pyplot as plt


def define_ha(settings, usafe_r):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton(name="Damped Oscillator")
    ha.variables = ["x", "y"]

    loc1 = ha.new_mode('loc1')

    loc1.a_matrix = np.array([[0.960659959352277, 0.194735414472060], [-0.194735414472060, 0.960659959352277]])
    loc1.c_vector = np.array([0.196702394590923, -0.019669801188631], dtype=float)
    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(loc1, error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([0.0, -1.0], -4))
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
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    s = HylaaSettings(step=0.2, max_time=10.0, disc_dyn=True, plot_settings=plot_settings)
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
    init_r = HyperRectangle([(-6, -5), (0, 1)])

    pv_object = run_hylaa(settings, init_r, None)
    # pv_object.compute_longest_ce()

    bdd_ce_object = BDD4CE(pv_object, equ_run=True, smt_mip='mip')
    bdd_graphs = bdd_ce_object.create_bdd_w_level_merge(level_merge=0, order='default')
    valid_exps, invalid_exps = bdd_graphs[0].generate_expressions()
    print(len(valid_exps), len(invalid_exps))
    Timers.print_stats()