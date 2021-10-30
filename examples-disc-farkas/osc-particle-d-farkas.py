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

    ha = LinearHybridAutomaton(name="Oscillating Particle")
    ha.variables = ["x", "y", "z"]

    loc1 = ha.new_mode('loc1')

    loc1.a_matrix = np.array([[0.722468865032875, -0.523371053120237, 0], [0.785056579680355, 0.696300312376864, 0], [0, 0, 0.930530895811206]])
    loc1.c_vector = np.array([0, 0.1, 0.03], dtype=float)
    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(loc1, error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([0, -1, 0], -0.5))
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
    plot_settings.plot_mode = PlotSettings.PLOT_MATLAB
    plot_settings.xdim = 2
    plot_settings.ydim = 1

    s = HylaaSettings(step=0.6, max_time=12.0, disc_dyn=True, plot_settings=plot_settings)
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
    # init_r = HyperRectangle([(-0.1, 0.1), (-1.0, -0.4), (1, 1.1)])
    init_r = HyperRectangle([(-0.1, 0.1), (-0.8, -0.4), (-1.07, -1)])
    # init_r = HyperRectangle([(0.1, 0.1), (-0.7813829650, -0.7813829650), (-1, -1)])  # smt longest ce
    # init_r = HyperRectangle([(0.0, 0.0), (-0.8, -0.8), (-1, -1)])  # milp longest ce

    # init_r = HyperRectangle([(0.0614, 0.0614), (-0.719, -0.719), (-1, -1)])  # sim 11110
    # init_r = HyperRectangle([(-0.089, -0.089), (-0.752, -0.752), (-1, -1)])  # sim 11101
    # init_r = HyperRectangle([(0.0276, 0.0276), (-0.694, -0.694), (-1.07, -1.07)])  # sim 11100
    # init_r = HyperRectangle([(0.1, 0.1), (-0.7469, -0.7469), (-1.07, -1.07)])  # sim 11010
    # init_r = HyperRectangle([(0.0279, 0.0279), (-0.562, -0.562), (-1.07, -1.07)])  # sim 11000
    # init_r = HyperRectangle([(-0.1, -0.1), (-0.7502, -0.7502), (-1, -1)])  # sim 01101
    # init_r = HyperRectangle([(-0.1, -0.1), (-0.601, -0.601), (-1, -1)])  # sim 01100
    # init_r = HyperRectangle([(0, 0), (-0.4578, -0.4578), (-1.07, -1.07)])  # sim 01000
    # 11110 - [0.0613950043, -0.7187912420, -1.0]
    # 11101 - [-0.1, -0.7700759978659456, -1.07]
    # 11100 - [0.02757849060634973,  -0.6941185448198288,  -1.07]
    # 11010 - [0.1, -0.7469429324390158, -1.07]
    # 11000 - [0.027911934534025906, -0.5624198571790866, -1.07]
    # 01101 - [-0.1, -0.7502076240, -1]
    # 01100 - [-0.1, -0.6010366411234954, -1]
    # 01000 - [0, -0.45779604927171486, -1.07]
    # 00000 - [0, -0.4, -1]

    pv_object = run_hylaa(settings, init_r, None)
    # ce_smt_object = CeSmt(pv_object)
    # ce_smt_object.compute_ce_w_regex(regex_str='11101')
    # ce_milp_object = CeMilp(pv_object)
    # ce_milp_object.compute_counterexample("Oscillator", regex='00000')
    bdd_ce_object = BDD4CE(pv_object, equ_run=True, smt_mip='mip')
    bdd_graphs = bdd_ce_object.create_bdd_w_level_merge(level_merge=0, order='random')
    valid_exps, invalid_exps = bdd_graphs[0].generate_expressions()
    print(len(valid_exps), len(invalid_exps))
    Timers.print_stats()
