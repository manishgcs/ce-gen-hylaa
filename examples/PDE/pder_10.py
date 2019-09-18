import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, HyperRectangle, LinearConstraint
from hylaa.engine import HylaaSettings, HylaaEngine
from hylaa.containers import PlotSettings
from hylaa.pv_container import PVObject
from hylaa.star import init_hr_to_star


def define_ha(settings, usafe_r):
    # x' = Ax + Bu + c

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "y1", "u1", "t", "stoptime"]

    loc1 = ha.new_mode('loc1')

    loc1.a_matrix = np.array([
        [-252.57, -106.6, 4.213, 13.09, -1.8157, 0.4038, 0.24558, -0.038164, -0.0034798, 0.0014783, 0, -51.94, 0, 0],
        [-106.6, -777.63, 63.509, 184.91, -26.094, 5.8051, 3.5304, -0.54862, -0.050025, 0.021252, 0, -11.124, 0, 0],
        [-4.213, -63.509, -25.034, -251.84, 21.751, -4.797, -2.9205, 0.45386, 0.041384, -0.017581, 0, -0.43288, 0, 0],
        [13.09, 184.91, 251.84, -634.37, 172.68, -39.239, -23.8, 3.698, 0.3372, -0.14325, 0, 1.3463, 0, 0],
        [-1.8157, -26.094, -21.751, 172.68, -645.44, 337.53, 175.84, -27.114, -2.474, 1.051, 0, -0.1867, 0, 0],
        [-0.4038, -5.8051, -4.797, 39.239, -337.53, -213.54, -248.03, 40.541, 3.6803, -1.5635, 0, -0.041519, 0, 0],
        [-0.24558, -3.5304, -2.9205, 23.8, -175.84, -248.03, -1671.7, 572.29, 47.319, -20.126, 0, -0.025252, 0, 0],
        [-0.038164, -0.54862, -0.45386, 3.698, -27.114, -40.541, -572.29, -438.0, -80.887, 33.914, 0, -0.0039241, 0, 0],
        [0.0034798, 0.050025, 0.041384, -0.3372, 2.474, 3.6803, 47.319, 80.887, -291.42, 259.2, 0, 0.00035781, 0, 0],
        [0.0014783, 0.021252, 0.017581, -0.14325, 1.051, 1.5636, 20.126, 33.914, -259.2, -1175.0, 0, 0.00015201, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=float)
    loc1.c_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=float)

    # loc1.inv_list.append(LinearConstraint([-51.94, -11.124, 0.43288, 1.3463, -0.1867, 0.041519, 0.025252, -0.0039241,
    #                                       -0.00035781, 0.000152, -1, 0, 0, 0], 0))
    # loc1.inv_list.append(LinearConstraint([51.94, 11.124, -0.43288, -1.3463, 0.1867, -0.041519, -0.025252, 0.0039241,
    #                                       0.00035781, -0.000152, 1, -0, -0, -0], -0))
    loc1.inv_list.append(LinearConstraint([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1], 0))

    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(loc1, error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(
            LinearConstraint([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], -0.002))
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

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
    plot_settings.xdim = 12
    plot_settings.ydim = 10

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
    init_r = HyperRectangle([(-0.0020389, -0.0011302), (-0.0004378, 0.0007629), (-0.0017922, -0.0003620),
                             (-0.0019804, -0.0005436), (-0.0008931, 0.0008771), (-0.0001891, 0.0014606),
                             (-0.0006451, 0.0012905), (-0.0002263, 0.0020114), (-0.0011494, 0.0012987),
                             (-0.0016059, 0.0017489), (0.0465752, 0.1101447), (0.5000000, 1.0000000),
                              (0, 0), (20.0, 20.0)])

    pv_object = run_hylaa(settings, init_r, None)
    # pv_object.compute_z3_counterexamples()
    # pv_object.compute_milp_counterexamples()
