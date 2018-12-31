
import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, HyperRectangle, LinearConstraint
from hylaa.star import init_hr_to_star
from hylaa.engine import HylaaSettings, HylaaEngine
from hylaa.containers import PlotSettings
from hylaa.new_pv_container import PVObject
from hylaa.timerutil import Timers
from hylaa.simutil import compute_simulation
import matplotlib.pyplot as plt

def define_ha(settings, usafe_r):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x", "y", "x1", "x2", "x3", "z"]

    loc1 = ha.new_mode('loc1')

    loc1.a_matrix = np.array(
        [[-2, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [5, 0, -5, 0, 0, 0], \
         [0, 0, 5, -5, 0, 0], [0, 0, 0, 5, -5, 0], [0, 0, 0, 0, 5, -5]], dtype=float)

    loc1.c_vector = np.array([1.4, -0.7, 0, 0, 0, 0], dtype=float)

    loc1.inv_list.append(LinearConstraint(
        [1, 0, 0, 0, 0, 0], 0.0))  # x <= 0
    loc1.inv_list.append(LinearConstraint(
        [-0.714286, -1, 0, 0, 0, 0], -0.0))  # y + 0.714286*x >=0

    loc2 = ha.new_mode('loc2')

    loc2.a_matrix = np.array(
        [[-2, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [5, 0, -5, 0, 0, 0], \
         [0, 0, 5, -5, 0, 0], [0, 0, 0, 5, -5, 0], [0, 0, 0, 0, 5, -5]], dtype=float)

    loc2.c_vector = np.array([-1.4, 0.7, 0, 0, 0, 0], dtype=float)

    loc2.inv_list.append(LinearConstraint(
        [1, 0, 0, 0, 0, 0], 0.0))  # x <= 0
    loc2.inv_list.append(LinearConstraint(
        [0.714286, 1, 0, 0, 0, 0], 0.0))  # y + 0.714286*x <=0

    loc3 = ha.new_mode('loc3')

    loc3.a_matrix = np.array(
        [[-2, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [5, 0, -5, 0, 0, 0], \
         [0, 0, 5, -5, 0, 0], [0, 0, 0, 5, -5, 0], [0, 0, 0, 0, 5, -5]], dtype=float)

    loc3.c_vector = np.array([1.4, -0.7, 0, 0, 0, 0], dtype=float)

    loc3.inv_list.append(LinearConstraint(
        [-1, 0, 0, 0, 0, 0], -0.0))  # x >= 0
    loc3.inv_list.append(LinearConstraint(
        [-0.714286, -1, 0, 0, 0, 0], -0.0))  # y + 0.714286*x >=0

    loc4 = ha.new_mode('loc4')

    loc4.a_matrix = np.array(
        [[-2, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [5, 0, -5, 0, 0, 0], \
         [0, 0, 5, -5, 0, 0], [0, 0, 0, 5, -5, 0], [0, 0, 0, 0, 5, -5]], dtype=float)

    loc4.c_vector = np.array([-1.4, 0.7, 0, 0, 0, 0], dtype=float)

    loc4.inv_list.append(LinearConstraint(
        [-1, 0, 0, 0, 0, 0], -0.0))  # x >= 0
    loc4.inv_list.append(LinearConstraint(
        [0.714286, 1, 0, 0, 0, 0], 0.0))  # y + 0.714286*x <=0

    trans3_4 = ha.new_transition(loc3, loc4)
    trans3_4.condition_list.append(LinearConstraint(
        [-1, 0, 0, 0, 0, 0], -0.0))  # x >= 0
    trans3_4.condition_list.append(LinearConstraint(
        [0.714286, 1, 0, 0, 0, 0], 0.0))  # y + 0.714286*x <= 0
    trans3_4.condition_list.append(LinearConstraint(
        [-0.714286, -1, 0, 0, 0, 0], -0.0))  # y + 0.714286*x >= 0

    trans4_2 = ha.new_transition(loc4, loc2)
    trans4_2.condition_list.append(LinearConstraint(
        [-1, 0, 0, 0, 0, 0], -0.0))  # x >= 0
    trans4_2.condition_list.append(LinearConstraint(
        [1, 0, 0, 0, 0, 0], 0.0))  # x <= 0
    trans4_2.condition_list.append(LinearConstraint(
        [0.714286, 1, 0, 0, 0, 0], 0.0))  # y + 0.714286*x <= 0

    trans2_1 = ha.new_transition(loc2, loc1)
    trans2_1.condition_list.append(LinearConstraint(
        [1, 0, 0, 0, 0, 0], 0.0))  # x <= 0
    trans2_1.condition_list.append(LinearConstraint(
        [0.714286, 1, 0, 0, 0, 0], 0.0))  # y + 0.714286*x <= 0
    trans2_1.condition_list.append(LinearConstraint(
        [-0.714286, -1, 0, 0, 0, 0], -0.0))  # y + 0.714286*x >= 0

    trans1_3 = ha.new_transition(loc1, loc3)
    trans1_3.condition_list.append(LinearConstraint(
        [-1, 0, 0, 0, 0, 0], -0.0))  # x >= 0
    trans1_3.condition_list.append(LinearConstraint(
        [1, 0, 0, 0, 0, 0], 0.0))  # x <= 0
    trans1_3.condition_list.append(LinearConstraint(
        [-0.714286, -1, 0, 0, 0, 0], -0.0))  # y + 0.714286*x >= 0

    error = ha.new_mode('_error')
    error.is_error = True
    trans4_error = ha.new_transition(loc4, error)
    trans3_error = ha.new_transition(loc3, error)
    trans2_error = ha.new_transition(loc2, error)

    usafe_set_constraint_list = []

    if usafe_r is None:
        #usafe_set_constraint_list.append(LinearConstraint([0, 0, 0, 0, 0, -1], -0.2))
        usafe_set_constraint_list.append(LinearConstraint([0, 0, 0, 0, 0, 1], 0.3))
        #usafe_set_constraint_list.append(LinearConstraint([0, -1, 0, 0, 0, 0], 0.3))


    #usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
    #for constraint in usafe_star.constraint_list:
    #    usafe_set_constraint_list.append(constraint)

    for constraint in usafe_set_constraint_list:
        trans4_error.condition_list.append(constraint)
        trans3_error.condition_list.append(constraint)
        trans2_error.condition_list.append(constraint)

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, y]
    rv = [(ha.modes['loc3'], init_r)]

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 1
    plot_settings.ydim = 5

    s = HylaaSettings(step=0.02, max_time=2.0, plot_settings=plot_settings)
    s.stop_when_error_reachable = False

    return s


def run_hylaa(settings, init_r, usafe_r):
    '''run hylaa with the given settings, returning the HylaaResult object.'''

    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)
    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    new_pv_object = PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)

    return new_pv_object


if __name__ == '__main__':
    #step = 0.08
    #max_time = 4.0
    settings = define_settings()
    init_r = HyperRectangle([(0.2, 0.3), (-0.1, 0.1), (0, 0), (0, 0), (0, 0), (0, 0)])

    #usafe_r = HyperRectangle([(0.4, 0.7), (-0.5, 0.1), (0.0, 0.6), (0.0, 0.6), (0, 0.6), (0, 0.6)])

    new_pv_object = run_hylaa(settings, init_r, None)

    longest_ce = new_pv_object.compute_longest_ce()
    depth_direction = np.identity(len(init_r.dims))
    deepest_ce = new_pv_object.compute_deepest_ce(depth_direction[5])
    robust_ce = new_pv_object.compute_robust_ce_new()
    new_pv_object.dump_path_constraints_for_milp()
    #loc3_a_matrix = np.array(
    #    [[-2, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [5, 0, -5, 0, 0, 0], \
    #     [0, 0, 5, -5, 0, 0], [0, 0, 0, 5, -5, 0], [0, 0, 0, 0, 5, -5]], dtype=float)

    #loc3_c_vector = np.array([1.4, -0.7, 0, 0, 0, 0], dtype=float)
    #deepest_simulation = compute_simulation(deepest_ce, loc3_a_matrix, loc3_c_vector, step, step*16 / step)
    #longest_simulation = compute_simulation(longest_ce, loc3_a_matrix, loc3_c_vector, step, step*16 / step)
    #sim_t = np.array(deepest_simulation).T
    #plt.plot(sim_t[1], sim_t[5], 'g-')
    #sim_t = np.array(longest_simulation).T
    #plt.plot(sim_t[1], sim_t[5], 'r-')
    #init_state_in_loc4 = deepest_simulation[len(deepest_simulation) - 1]
    #loc4_a_matrix = np.array(
    #    [[-2, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [5, 0, -5, 0, 0, 0], \
    #     [0, 0, 5, -5, 0, 0], [0, 0, 0, 5, -5, 0], [0, 0, 0, 0, 5, -5]], dtype=float)

    #loc4_c_vector = np.array([-1.4, 0.7, 0, 0, 0, 0], dtype=float)
    #deepest_simulation = compute_simulation(init_state_in_loc4, loc4_a_matrix, loc4_c_vector, step, step*5 / step)
    #sim_t = np.array(deepest_simulation).T
    #plt.plot(sim_t[1], sim_t[5], 'g-')
    #init_state_in_loc4 = longest_simulation[len(longest_simulation) - 1]
    #longest_simulation = compute_simulation(init_state_in_loc4, loc4_a_matrix, loc4_c_vector, step, step*5 / step)
    #sim_t = np.array(longest_simulation).T
    #plt.plot(sim_t[1], sim_t[5], 'r-')

    #loc2_a_matrix = np.array(
    #    [[-2, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [5, 0, -5, 0, 0, 0], \
    #     [0, 0, 5, -5, 0, 0], [0, 0, 0, 5, -5, 0], [0, 0, 0, 0, 5, -5]], dtype=float)

    #loc2_c_vector = np.array([-1.4, 0.7, 0, 0, 0, 0], dtype=float)
    #init_state_in_loc2 = deepest_simulation[len(deepest_simulation) - 1]
    #deepest_simulation = compute_simulation(init_state_in_loc2, loc2_a_matrix, loc2_c_vector, step, step * 16 / step)
    #sim_t = np.array(deepest_simulation).T
    #plt.plot(sim_t[1], sim_t[5], 'g-')
    #plt.show()
    Timers.print_stats()
