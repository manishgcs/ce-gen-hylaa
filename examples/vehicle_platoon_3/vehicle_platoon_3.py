import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, HyperRectangle, LinearConstraint
from hylaa.star import init_hr_to_star
from hylaa.engine import HylaaSettings, HylaaEngine
from hylaa.containers import PlotSettings
from hylaa.timerutil import Timers
from hylaa.pv_container import PVObject
import matplotlib.pyplot as plt


def define_ha(settings, usafe_r):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["e1", "e1p", "a1", "e2", "e2p", "a2", "e3", "e3p", "a3", "t"]

    loc1 = ha.new_mode('loc1')

    loc1.a_matrix = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                              [1.605, 4.868, -3.5754, -0.8198, 0.427, -0.045, -0.1942, 0.3626, -0.0946, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, -1, 0, 0, 0, 0],
                              [0.8718, 3.814, -0.0754, 1.1936, 3.6258, -3.2396, -0.595, 0.1294, -0.0796, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, -1, 0],
                              [0.7132, 3.573, -0.0964, 0.8472, 3.2568, -0.0876, 1.2726, 3.072, -3.1356, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    loc1.c_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float)

    loc2 = ha.new_mode('loc2')

    loc2.a_matrix = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                              [1.605, 4.868, -3.5754, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 1.1936, 3.6258, -3.2396, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, -1, 0],
                              [0.7132, 3.573, -0.0964, 0.8472, 3.2568, -0.0876, 1.2726, 3.072, -3.1356, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    loc2.c_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float)

    loc3 = ha.new_mode('loc3')

    loc3.a_matrix = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                              [1.605, 4.868, -3.5754, -0.8198, 0.427, -0.045, -0.1942, 0.3626, -0.0946, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, -1, 0, 0, 0, 0],
                              [0.8718, 3.814, -0.0754, 1.1936, 3.6258, -3.2396, -0.595, 0.1294, -0.0796, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, -1, 0],
                              [0.7132, 3.573, -0.0964, 0.8472, 3.2568, -0.0876, 1.2726, 3.072, -3.1356, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    loc3.c_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float)

    loc4 = ha.new_mode('loc4')

    loc4.a_matrix = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                              [1.605, 4.868, -3.5754, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, 1.1936, 3.6258, -3.2396, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, -1, 0],
                              [0.7132, 3.573, -0.0964, 0.8472, 3.2568, -0.0876, 1.2726, 3.072, -3.1356, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
    loc4.c_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float)

    trans1_2 = ha.new_transition(loc1, loc2)
    trans1_2.condition_list.append(LinearConstraint(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1], -3.0))  # t >= 3

    trans2_3 = ha.new_transition(loc2, loc3)
    trans2_3.condition_list.append(LinearConstraint(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1], -4.0))  # t >= 5

    trans3_4 = ha.new_transition(loc3, loc4)
    trans3_4.condition_list.append(LinearConstraint(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1], -6.0))  # t >= 5

    error = ha.new_mode('_error')
    error.is_error = True

    trans1_err = ha.new_transition(loc1, error)
    trans2_err = ha.new_transition(loc2, error)
    trans3_err = ha.new_transition(loc3, error)
    trans4_err = ha.new_transition(loc4, error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        # Create the list of constraints defining unsafe set if hyperrectangle representation is not given
        usafe_set_constraint_list.append(LinearConstraint([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0], -1.7)) # e1 >= 1.7
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    for constraint in usafe_set_constraint_list:
        trans1_err.condition_list.append(constraint)
        trans2_err.condition_list.append(constraint)
        trans3_err.condition_list.append(constraint)
        trans4_err.condition_list.append(constraint)

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    rv = []
    rv.append((ha.modes['loc1'], init_r))

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    # s = HylaaSettings(step=0.04, max_time=5.0, plot_settings=plot_settings)
    s = HylaaSettings(step=0.5, max_time=5.0, plot_settings=plot_settings)
    s.stop_when_error_reachable = False

    return s


def run_hylaa(settings, init_r, usafe_r):
    '''Runs hylaa with the given settings, returning the HylaaResult object.'''

    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)
    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1),
                             (0.9, 1.1), (0.9, 1.1), (0, 0)])

    usafe_r = None
    pv_object = run_hylaa(settings, init_r, usafe_r)
    pv_object.compute_longest_ce()

    depth_direction = np.identity(len(init_r.dims))
    deepest_ce = pv_object.compute_deepest_ce(depth_direction[1])
    Timers.print_stats()
