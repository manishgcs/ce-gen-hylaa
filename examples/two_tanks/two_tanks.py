
import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint, HyperRectangle
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.plotutil import PlotSettings
from hylaa.star import init_hr_to_star
from hylaa.timerutil import Timers
from hylaa.pv_container import PVObject


def define_ha(settings, usafe_r=None):
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2"]

    # loc1 == off_off
    loc1 = ha.new_mode('loc1')
    loc1.a_matrix = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float)
    loc1.c_vector = np.array([-2.0, 0.0], dtype=float)
    loc1.inv_list.append(LinearConstraint([-1.0, 0.0], 1.0)) # x1 >= -1
    loc1.inv_list.append(LinearConstraint([0.0, 1.0], 1.0))  # x2 <= 1

    # loc2 == on_off
    loc2 = ha.new_mode('loc2')
    loc2.a_matrix = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float)
    loc2.c_vector = np.array([3.0, 0.0], dtype=float)
    loc2.inv_list.append(LinearConstraint([0.0, 1.0], 1.0))  # x2 <= 1

    # loc3 == off_on
    loc3 = ha.new_mode('loc3')
    loc3.a_matrix = np.array([[-1.0, 0.0], [1.0, -1.0]], dtype=float)
    loc3.c_vector = np.array([-2.0, -5.0], dtype=float)
    loc3.inv_list.append(LinearConstraint([-1.0, 0.0], 1.0))  # x1 >= -1
    loc3.inv_list.append(LinearConstraint([0.0, -1.0], -0.0))  # x2 >= 0

    # loc4 == on_on
    loc4 = ha.new_mode('loc4')
    loc4.a_matrix = np.array([[-1.0, 0.0], [1.0, -1.0]], dtype=float)
    loc4.c_vector = np.array([3.0, -5.0], dtype=float)
    loc4.inv_list.append(LinearConstraint([1.0, 0.0], 1.0))  # x1 <= 1
    loc4.inv_list.append(LinearConstraint([0.0, -1.0], -0.0)) # x2 >= 0

    trans1_2 = ha.new_transition(loc1, loc2)
    trans1_2.condition_list.append(LinearConstraint([1.0, 0.0], -1.0))  # x1 <= -1
    trans1_2.condition_list.append(LinearConstraint([-1.0, 0.0], 1.0))  # x1 >= -1

    trans2_3 = ha.new_transition(loc2, loc3)
    trans2_3.condition_list.append(LinearConstraint([0.0, 1.0], 1.0))  # x2 <= 1
    trans2_3.condition_list.append(LinearConstraint([0.0, -1.0], -1.0))  # x2 >= 1

    trans3_1 = ha.new_transition(loc3, loc1)
    trans3_1.condition_list.append(LinearConstraint([0.0, 1.0], 0.0))  # x2 <= 0
    trans3_1.condition_list.append(LinearConstraint([0.0, -1.0], -0.0))  # x2 >= 0

    trans1_3 = ha.new_transition(loc1, loc3)
    trans1_3.condition_list.append(LinearConstraint([0.0, 1.0], 1.0))  # x2 <= 1
    trans1_3.condition_list.append(LinearConstraint([0.0, -1.0], -1.0))  # x2 >= 1

    trans3_4 = ha.new_transition(loc3, loc4)
    trans3_4.condition_list.append(LinearConstraint([1.0, 0.0], -1.0))  # x1 <= -1
    trans3_4.condition_list.append(LinearConstraint([-1.0, 0.0], 1.0))  # x1 >= -1

    trans4_3 = ha.new_transition(loc4, loc3)
    trans4_3.condition_list.append(LinearConstraint([1.0, 0.0], 1.0))  # x1 <= 1
    trans4_3.condition_list.append(LinearConstraint([-1.0, 0.0], -1.0))  # x1 >= 1

    trans4_2 = ha.new_transition(loc4, loc2)
    trans4_2.condition_list.append(LinearConstraint([0.0, 1.0], 0.0))  # x2 <= 0
    trans4_2.condition_list.append(LinearConstraint([0.0, -1.0], -0.0))  # x2 >= 0

    error = ha.new_mode(('_error'))
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([0.0, 1.0], -0.2))
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    trans1_error = ha.new_transition(loc1, error)
    for constraint in usafe_set_constraint_list:
        trans1_error.condition_list.append(constraint)

    trans2_error = ha.new_transition(loc3, error)
    for constraint in usafe_set_constraint_list:
        trans2_error.condition_list.append(constraint)

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, v]
    rv = []

    rv.append((ha.modes['loc3'], init_r))

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    settings = HylaaSettings(step=0.01, max_time=2.0, plot_settings=plot_settings)
    settings.stop_when_error_reachable = False

    return settings


def run_hylaa(settings, init_r, usafe_r):
    '''Runs hylaa with the given settings, returning the HylaaResult object.'''

    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)

    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(1.5, 2.5), (1, 1.1)])
    # usafe_r = HyperRectangle([(0, 0.5), (-0.2, 0.2)]) #ToCheck for Depth

    # usafe_r = HyperRectangle([(0.5, 1.0), (-0.2, 0.1)])  # Small orig
    # usafe_r = HyperRectangle([(0.5, 0.7), (-0.2, 0.1)]) #Small checking in MILP
    usafe_r = HyperRectangle([(0, 1.1), (-0.3, 0.3)])  # Medium
    # usafe_r = HyperRectangle([(0, 1.9), (-0.3, 0.7)])  # Large:

    pv_object = run_hylaa(settings, init_r, usafe_r)
    # longest_ce = pv_object.compute_longest_ce()
    # depth_direction = np.identity(len(init_r.dims))
    # deepest_ce = pv_object.compute_deepest_ce(depth_direction[0])
    # robust_ce = pv_object.compute_robust_ce_new()
    pv_object.compute_milp_counterexample('Tanks')
    pv_object.compute_z3_counterexamples()
    # z3_counter_examples = pv_object.compute_counter_examples_using_z3(2)
    Timers.print_stats()
