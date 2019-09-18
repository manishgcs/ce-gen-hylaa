import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, HyperRectangle, LinearConstraint
from hylaa.star import init_hr_to_star
from hylaa.engine import HylaaSettings, HylaaEngine
from hylaa.containers import PlotSettings
from hylaa.simutil import compute_simulation
from hylaa.timerutil import Timers
from hylaa.pv_container import PVObject
import matplotlib.pyplot as plt


def define_ha(settings, usafe_r):
    # x' = Ax + Bu + c

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "y1", "y2", "y3", "u1", "u2", "u3",
                    "t", "stoptime"]

    loc1 = ha.new_mode('loc1')

    loc1.a_matrix = np.array([
        [- 3.9874e-8, 0.77508, - 0.000042976, 1.1479e-7, - 7.09e-9, - 1.653e-7, 0.000082292, 1.2158e-7, - 0.000019112,
         1.4896e-6, 0, 0, 0, - 0.000067832, - 1.5818e-8, - 4.4341e-6, 0, 0],
        [- 0.77508, - 0.0077527, 0.010393, - 0.000091911, 3.0412e-6, 0.000067451, - 0.030171, - 0.000093795, 0.0080792,
         - 0.00062151, 0, 0, 0, - 0.029896, - 7.046e-6, - 0.0021408, 0, 0],
        [0.000042976, 0.010393, - 0.019925, 1.992, - 9.64e-6, - 0.00020176, 0.074349, 0.00047222, - 0.02434, 0.0018562,
         0, 0, 0, 0.025889, 5.692e-6, 0.0017739, 0, 0],
        [1.1344e-7, 0.000091528, - 1.992, - 4.721e-7, 6.8864e-8, 1.3712e-6, - 0.00069225, 4.1564e-7, 0.00016249,
         - 0.000011465, 0, 0, 0, 0.00012603, 6.3733e-9, - 8.2928e-6, 0, 0],
        [2.267e-9, 3.4178e-7, 1.4523e-6, - 3.9616e-8, - 2.0742e-7, 8.4808, 0.00019256, - 8.1914e-6, - 0.00027239,
         4.7e-6, 0, 0, 0, 1.6594e-6, 0.000049597, - 5.5481e-6, 0, 0],
        [8.4062e-8, 0.00002251, - 0.000015118, - 8.7922e-7, - 8.4808, - 0.084954, 0.0020235, 0.00035617, - 0.00040077,
         0.000061347, 0, 0, 0, 0.000050285, 0.031956, - 8.8146e-6, 0, 0],
        [- 0.000082305, - 0.030181, 0.074387, 0.00069032, - 0.00015437, 0.00051015, - 0.38083, - 37.979, 0.7873,
         - 0.060466, 0, 0, 0, - 0.06356, - 0.00033453, - 0.004295, 0, 0],
        [4.9915e-7, 0.00023226, - 0.00081483, - 3.7771e-6, 6.7246e-6, - 0.00034414, 37.981, - 0.000035297, 0.00053662,
         - 0.00070952, 0, 0, 0, 0.00046227, 4.5013e-6, - 0.000403, 0, 0],
        [7.2178e-6, 0.0017, - 0.00019482, - 0.00011165, 0.00019205, 0.00079526, - 0.6624, - 0.003324, - 0.095564,
         - 9.226, 0, 0, 0, 0.0067388, - 0.00010737, - 0.029739, 0, 0],
        [- 6.128e-7, - 0.000159060, 0.000098378, 7.5464e-6, - 2.7468e-6, - 0.00014204, 0.050261, 0.00077848, 9.2327,
         - 0.00016399, 0, 0, 0, - 0.00049267, 0.000018473, 0.0011629, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=float)
    loc1.c_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=float)
    # loc1.inv_list.append(LinearConstraint([0.000067947, - 0.029964, 0.025941, - 0.00012623, 6.5319e-6, 0.00013057,
    #                                       - 0.06363, - 0.00012503, 0.01654, - 0.0012343, -1, 0, 0, 0, 0, 0, 0, 0], 0))
    # loc1.inv_list.append(LinearConstraint([-0.000067947, 0.029964, -0.025941, 0.00012623, -6.5319e-6, -0.00013057,
    #                                      0.06363, 0.00012503, -0.01654, 0.0012343, 1, -0, -0, -0, -0, -0, -0, -0], -0))
    # loc1.inv_list.append(LinearConstraint(
    #    [5.5487e-9, - 3.8275e-6, 3.2968e-6, - 4.5505e-8, - 0.000043816, 0.031956, - 0.00020452, - 8.8804e-6,
    #     - 0.0001039, 0.000015029, 0, -1, 0, 0, 0, 0, 0, 0], 0))
    # loc1.inv_list.append(LinearConstraint(
    #    [-5.5487e-9, 3.8275e-6, -3.2968e-6, 4.5505e-8, 0.000043816, -0.031956, 0.00020452, 8.8804e-6, 0.0001039,
    #     -0.000015029, -0, 1, -0, -0, -0, -0, -0, -0], -0))
    # loc1.inv_list.append(LinearConstraint(
    #    [2.0132e-6, - 0.00075046, 0.00065699, 4.4664e-6, - 0.000023041, 0.000011699, - 0.0031187, - 0.00060035,
    #     - 0.025617, 0.00026762, 0, 0, -1, 0, 0, 0, 0, 0], 0))
    # loc1.inv_list.append(LinearConstraint(
    #    [-2.0132e-6, 0.00075046, -0.00065699, -4.4664e-6, 0.000023041, -0.000011699, 0.0031187, 0.00060035,
    #     0.025617, -0.00026762, -0, -0, 1, -0, -0, -0, -0, -0], -0))
    loc1.inv_list.append(LinearConstraint([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1], 0))
    # loc1.inv_list.append(LinearConstraint([-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -1, 1], -0))

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], -0.002))
        usafe_set_constraint_list.append(LinearConstraint([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.004))
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
    rv = []
    rv.append((ha.modes['loc1'], init_r))

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 16
    plot_settings.ydim = 0

    s = HylaaSettings(step=0.2, max_time=20.0, plot_settings=plot_settings)
    s.stop_when_error_reachable = False

    return s


def run_hylaa(settings, init_r, usafe_r):
    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)
    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(-0.0000046, 0.0000046), (-0.0000064, 0.0000064), (-0.0000073, 0.0000073),
                             (-0.0000119, 0.0000119), (-0.0000316, 0.0000316), (-0.0000065, 0.0000065),
                             (-0.0000169, 0.0000169), (-0.0002037, 0.0002037), (-0.0000173, 0.0000173),
                             (-0.0000508, 0.0000508), (-0.0000018, 0.0000018), (-0.0000002, 0.0000002),
                             (-0.0000006, 0.0000006), (0.0000000, 0.1000000), (00.8000000, 1.0000000),
                             (0.9000000, 1.0000000), (0, 0), (20.0, 20.0)])

    pv_object = run_hylaa(settings, init_r, None)
    pv_object.compute_z3_counterexamples()
    pv_object.compute_milp_counterexamples('ISS')

    # c_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=float)
    # milp_ce = np.array([-4.6e-06, 6.4e-06, 7.3e-06, 1.19e-05, -3.16e-05, -6.5e-06, 1.69e-05, -0.0002037, 1.73e-05,
    #                    -5.08e-05, 0, 0, 0, 0, 0.8, 0.90818, 0, 20])
    # smt_ce = np.array([-0.0000046, -0.0000064, -0.0000073, -0.0000119, 0.0000316, -0.0000065, 0.0000169, -0.0002037,
    #                   0.0000173, 0.0000508, 0, 0, 0, 0.0004226513, 0.8, 0.9, 0, 20])
    # simulation = compute_simulation(milp_ce, a_matrix, c_vector, 0.2, 20 / 0.2)
    # sim = np.array(simulation).T
    # plt.plot(sim[16], sim[0], 'r^--')
    # simulation = compute_simulation(smt_ce, a_matrix, c_vector, 0.2, 20 / 0.2)
    # sim = np.array(simulation).T
    # plt.plot(sim[16], sim[0], 'b-*')
    # plt.show()
    Timers.print_stats()