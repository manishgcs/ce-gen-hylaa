

import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint, HyperRectangle
from hylaa.engine import HylaaSettings
from hylaa.star import init_hr_to_star
from hylaa.engine import HylaaEngine
from hylaa.plotutil import PlotSettings
from hylaa.pv_container import PVObject
from hylaa.timerutil import Timers
from hylaa.simutil import compute_simulation
import matplotlib.pyplot as plt


def define_ha(settings, usafe_r):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    cap_d = 0.4
    cap_t = 0.000025
    L = 0.00008  # L = 80*10e-6
    one_by_L = 12500
    one_by_L_by_3 = 4166.67
    R = 30
    one_by_C = 1000000  # C = 1e-6
    one_by_RC = 33333.33
    v_in = 100   #u

    ha = LinearHybridAutomaton()
    ha.variables = ["ilm", "il", "vc", "u", "t"]

    loc1 = ha.new_mode('loc1')

    loc1.a_matrix = np.array(
        [[0, 0, 0, 0, 0], [0, 0, -one_by_L, 0, 0], [0, one_by_C, -one_by_RC, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]], dtype=float)

    loc1.c_vector = np.array([one_by_L*v_in, L*v_in, 0, 0, 1], dtype=float)

    loc1.inv_list.append(LinearConstraint(
        [0, 0, 0, 0, 1], cap_d*cap_t))  # t <= DT
    loc1.inv_list.append(LinearConstraint(
        [0, -1, 0, 0, 0], -0.0))  # il >= 0
    loc1.inv_list.append(LinearConstraint(
        [-1, 0, 0, 0, 0], -0.0))  # ilm >= 0

    loc2 = ha.new_mode('loc2')

    loc2.a_matrix = np.array(
        [[0, 0, 0, 0, 0], [0, 0, -one_by_L, 0, 0], [0, one_by_C, -one_by_RC, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]], dtype=float)

    loc2.c_vector = np.array([-one_by_L_by_3, 0, 0, 0, 1], dtype=float)

    loc2.inv_list.append(LinearConstraint(
        [0, 0, 0, 0, 1], cap_t))  # t <= T
    loc2.inv_list.append(LinearConstraint(
        [0, -1, 0, 0, 0], -0.0))  # il >= 0
    loc2.inv_list.append(LinearConstraint(
        [-1, 0, 0, 0, 0], -0.0))  # ilm >= 0

    loc3 = ha.new_mode('loc3')

    loc3.a_matrix = np.array(
        [[0, 0, 0, 0, 0], [0, 0, -one_by_L, 0, 0], [0, one_by_C, -one_by_RC, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]], dtype=float)

    loc3.c_vector = np.array([0, 0, 0, 0, 1], dtype=float)

    loc3.inv_list.append(LinearConstraint(
        [0, 0, 0, 0, 1], cap_t))  # t <= T
    loc3.inv_list.append(LinearConstraint(
        [0, -1, 0, 0, 0], -0.0))  # il >= 0
    loc3.inv_list.append(LinearConstraint(
        [1, 0, 0, 0, 0], 0.0))  # ilm <= 0

    loc4 = ha.new_mode('loc4')

    loc4.a_matrix = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -one_by_RC, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]], dtype=float)

    loc4.c_vector = np.array([0, 0, 0, 0, 1], dtype=float)

    loc4.inv_list.append(LinearConstraint(
        [0, 0, 0, 0, 1], cap_t))  # t <= DT
    loc4.inv_list.append(LinearConstraint(
        [0, 1, 0, 0, 0], 0.0))  # il <= 0
    loc4.inv_list.append(LinearConstraint(
        [1, 0, 0, 0, 0], 0.0))  # ilm <= 0

    loc5 = ha.new_mode('loc5')

    loc5.a_matrix = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -one_by_RC, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]], dtype=float)

    loc5.c_vector = np.array([-one_by_L_by_3, 0, 0, 0, 1], dtype=float)

    loc5.inv_list.append(LinearConstraint(
        [0, 0, 0, 0, 1], cap_t))  # t <= T
    loc5.inv_list.append(LinearConstraint(
        [0, 1, 0, 0, 0], 0.0))  # il <= 0
    loc5.inv_list.append(LinearConstraint(
        [-1, 0, 0, 0, 0], -0.0))  # ilm >= 0

    loc6 = ha.new_mode('loc6')

    loc6.a_matrix = np.array(
        [[0, 0, 0, 0, 0], [0, 0, -one_by_L, 0, 0], [0, one_by_C, -one_by_RC, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]], dtype=float)

    loc6.c_vector = np.array([one_by_L * v_in, L * v_in, 0, 0, 1], dtype=float)

    loc6.inv_list.append(LinearConstraint(
        [0, 0, 0, 0, 1], cap_t + (cap_d * cap_t)))  # t <= T + DT
    loc6.inv_list.append(LinearConstraint(
        [0, -1, 0, 0, 0], -0.0))  # il >= 0
    loc6.inv_list.append(LinearConstraint(
        [-1, 0, 0, 0, 0], -0.0))  # ilm >= 0

    trans1_2 = ha.new_transition(loc1, loc2)
    trans1_2.condition_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, -1], -cap_d * cap_t))  # t >= DT

    trans2_3 = ha.new_transition(loc2, loc3)
    trans2_3.condition_list.append(LinearConstraint([1.0, 0.0, 0.0, 0.0, 0], 0.0))  # ilm <= 0

    trans2_5 = ha.new_transition(loc2, loc5)
    trans2_5.condition_list.append(LinearConstraint([0.0, 1.0, 0.0, 0.0, 0.0], 0.0))  # il <= 0

    trans3_4 = ha.new_transition(loc3, loc4)
    trans3_4.condition_list.append(LinearConstraint([0.0, 1.0, 0.0, 0.0, 0.0], 0.0))  # il <= 0

    trans5_4 = ha.new_transition(loc5, loc4)
    trans5_4.condition_list.append(LinearConstraint([1.0, 0.0, 0.0, 0.0, 0.0], 0.0))  # ilm <= 0

    trans2_6 = ha.new_transition(loc2, loc6)
    trans2_6.condition_list.append(LinearConstraint([0.0, 0.0, 0, 0.0, -1], -cap_t))  # t >= T

    trans3_6 = ha.new_transition(loc3, loc6)
    trans3_6.condition_list.append(LinearConstraint([0.0, 0.0, 0, 0.0, -1], -cap_t))  # t >= T

    trans4_6 = ha.new_transition(loc4, loc6)
    trans4_6.condition_list.append(LinearConstraint([0.0, 0.0, 0, 0.0, -1], -cap_t))  # t >= T

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:
        # usafe_set_constraint_list.append(LinearConstraint([0.0, 0.0, -1.0, 0.0, 0.0], -2.50)) # vc >= 2.5
        usafe_set_constraint_list.append(LinearConstraint([0.0, 0.0, -1.0, 0.0, 0.0], -2.20))  # vc >= 2.2
        # usafe_set_constraint_list.append(LinearConstraint([0.0, 0.0, -1.0, 0.0, 0.0], -2.00))  # vc >= 2.0
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    trans1_e = ha.new_transition(loc1, error)
    trans2_e = ha.new_transition(loc2, error)
    trans5_e = ha.new_transition(loc5, error)
    trans6_e = ha.new_transition(loc6, error)
    for pred in usafe_set_constraint_list:
        trans1_e.condition_list.append(pred.clone())
        trans2_e.condition_list.append(pred.clone())
        trans5_e.condition_list.append(pred.clone())
        trans6_e.condition_list.append(pred.clone())

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
    plot_settings.xdim = 1
    plot_settings.ydim = 2

    settings = HylaaSettings(step=0.000001, max_time=0.0001, plot_settings=plot_settings)
    settings.stop_when_error_reachable = False

    return settings


def compute_simulation_py(robust_ce, longest_ce):
    cap_d = 0.4
    cap_t = 0.000025
    L = 0.00008  # L = 80*10e-6
    one_by_L = 12500
    one_by_L_by_3 = 4166.67
    R = 30
    one_by_C = 1000000  # C = 1e-6
    one_by_RC = 33333.33
    v_in = 100  # u
    step = 0.000001
    loc1_a_matrix = np.array(
        [[0, 0, 0, 0, 0], [0, 0, -one_by_L, 0, 0], [0, one_by_C, -one_by_RC, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]], dtype=float)

    loc1_c_vector = np.array([one_by_L * v_in, L * v_in, 0, 0, 1], dtype=float)
    robust_simulation = compute_simulation(robust_ce, loc1_a_matrix, loc1_c_vector, step, step*11 / step)
    sim_t = np.array(robust_simulation).T
    plt.plot(sim_t[1], sim_t[2], 'r--', linewidth='2')
    init_state_in_loc2 = robust_simulation[len(robust_simulation) - 1]
    loc2_a_matrix = np.array(
        [[0, 0, 0, 0, 0], [0, 0, -one_by_L, 0, 0], [0, one_by_C, -one_by_RC, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]], dtype=float)

    loc2_c_vector = np.array([-one_by_L_by_3, 0, 0, 0, 1], dtype=float)
    robust_simulation = compute_simulation(init_state_in_loc2, loc2_a_matrix, loc2_c_vector, step, step * 5 / step)
    sim_t = np.array(robust_simulation).T
    plt.plot(sim_t[1], sim_t[2], 'r--', linewidth='2')
    init_state_in_loc5 = robust_simulation[len(robust_simulation) - 1]
    loc5_a_matrix = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -one_by_RC, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]], dtype=float)

    loc5_c_vector = np.array([-one_by_L_by_3, 0, 0, 0, 1], dtype=float)
    robust_simulation = compute_simulation(init_state_in_loc5, loc5_a_matrix, loc5_c_vector, step, step * 2 / step)
    sim_t = np.array(robust_simulation).T
    plt.plot(sim_t[1], sim_t[2], 'r--', linewidth='2')

    longest_simulation = compute_simulation(longest_ce, loc1_a_matrix, loc1_c_vector, step, step * 11 / step)
    sim_t = np.array(longest_simulation).T
    plt.plot(sim_t[1], sim_t[2], 'b--', linewidth='2')
    init_state_in_loc2 = longest_simulation[len(longest_simulation) - 1]
    longest_simulation = compute_simulation(init_state_in_loc2, loc2_a_matrix, loc2_c_vector, step, step * 5 / step)
    sim_t = np.array(longest_simulation).T
    plt.plot(sim_t[1], sim_t[2], 'b--', linewidth='2')
    init_state_in_loc5 = longest_simulation[len(longest_simulation) - 1]
    longest_simulation = compute_simulation(init_state_in_loc5, loc5_a_matrix, loc5_c_vector, step, step * 2 / step)
    sim_t = np.array(longest_simulation).T
    plt.plot(sim_t[1], sim_t[2], 'b--', linewidth='2')
    plt.show()


def run_hylaa(settings, init_r, usafe_r):
    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)

    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(0.0, 0.4), (0.0, 0.4), (0.0, 0.4), (0, 0), (0.0, 0.0)])
    pv_object = run_hylaa(settings, init_r, None)
    longest_ce = pv_object.compute_longest_ce()
    depth_direction = np.identity(len(init_r.dims))
    deepest_ce = pv_object.compute_deepest_ce(depth_direction[2])
    robust_ce = pv_object.compute_robust_ce_new()
    compute_simulation_py(robust_ce, longest_ce)
    Timers.print_stats()

