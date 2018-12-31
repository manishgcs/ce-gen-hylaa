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
from hylaa.post_verif_container import PostVerificationObject
from hylaa.new_pv_container import PVObject
from hylaa.timerutil import Timers
from hylaa.simutil import compute_simulation
import matplotlib.pyplot as plt

from z3 import *
from matplotlib.path import Path
import matplotlib.patches as patches
usafe_star = None


def define_ha(settings, usafe_r):
    #x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x", "y"]

    loc1 = ha.new_mode('loc1')

    loc1.a_matrix = np.array([[-0.1, 1], [-1, -0.1]])
    loc1.c_vector = np.array([0, 0], dtype=float)
    #loc1.set_dynamics(a_matrix, c_vector)

    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(loc1, error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0], 2))
        usafe_set_constraint_list.append(LinearConstraint([1.0, 0.0], 2))
        usafe_set_constraint_list.append(LinearConstraint([0.0, -1.0], -1))
        usafe_set_constraint_list.append(LinearConstraint([0.0, 1.0], 6))
    else:
        global usafe_star
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

    s = HylaaSettings(step=0.2, max_time=20.0, plot_settings=plot_settings)
    s.stop_when_error_reachable = False
    
    return s


def run_hylaa(settings, init_r, usafe_r):

    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)
    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    #post_verif_object = PostVerificationObject(settings, ha, init, usafe_set_constraint_list, error_stars)

    #long_ce_direction = np.ones(len(init_r.dims))
    #post_verif_object.compute_longest_ce(long_ce_direction)
    #depth_direction = np.identity(len(init_r.dims))
    #post_verif_object.compute_deepest_ce(depth_direction[1])

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(-6, -5), (0, 1)])

    #usafe_r = HyperRectangle([(-1.5, 1.5), (4, 6)])  # Small
    usafe_r = HyperRectangle([(-2, 2), (1, 5)])  # milp
    #usafe_r = HyperRectangle([(-3, 2), (1, 5)]) #Medium
    #usafe_r = HyperRectangle([(-4, 2), (-1, 6)]) #Large

    new_pv_object = run_hylaa(settings, init_r, usafe_r)
    new_pv_object.compute_longest_ce()
    depth_direction = np.identity(len(init_r.dims))
    new_pv_object.compute_deepest_ce(depth_direction[1])
    robust_pt = new_pv_object.compute_robust_ce_new()
    #milp_pt = np.array([-5.6315, 0.0749016])
    #milp_pt = np.array([-5.0, 0.398188])
    milp_pt = np.array([-5.51916, 0.590218])
    #robust_pt = milp_pt
    new_pv_object.dump_path_constraints_in_a_file()
    z3_counter_examples = []
    z3_counter_examples = new_pv_object.compute_counter_examples_using_z3(19)
    new_pv_object.dump_path_constraints_for_milp()
    a_matrix = np.array([[-0.1, 1], [-1, -0.1]])
    c_vector = np.array([0, 0], dtype=float)
    simulations = []
    for ce in z3_counter_examples:
        simulation = compute_simulation(robust_pt, a_matrix, c_vector, 0.2, 20/0.2)
        x, y = np.array(simulation).T
        plt.plot(x, y, 'r*')
    global usafe_star
    verts = usafe_star.verts()
    print verts
    x, y = np.array(verts).T
    #rob_x, rob_y = np.array(robust_pts).T
    plt.plot(x, y, 'r-', robust_pt[0], robust_pt[1], 'r.')
    plt.show()
    Timers.print_stats()