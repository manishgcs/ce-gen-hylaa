
import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint, HyperRectangle
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.plotutil import PlotSettings
from hylaa.star import init_hr_to_star
from hylaa.new_pv_container import PVObject
from hylaa.simutil import compute_simulation
import matplotlib.pyplot as plt


def define_ha(settings, usafe_r=None):
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x", "y", "vx", "vy", "t"]

    p2 = ha.new_mode('P2')
    p2.a_matrix = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                            [-0.057599765881773, 0.0002009598965197660, -2.89995083970656, 0.00877200894463775, 0],
                            [-0.000174031357370456, -0.0665123984901026, 0.00875351105536225, -2.90300269286856, 0],
                            [0, 0, 0, 0, 0]], dtype=float)
    p2.c_vector = np.array([0, 0, 0, 0, 1], dtype=float)
    p2.inv_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, 1.0], 125))  # t <= 125
    p2.inv_list.append(LinearConstraint([1.0, 0.0, 0.0, 0.0, 0.0], -100))  # x <= -100

    p3 = ha.new_mode('P3')
    p3.a_matrix = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 1],
                            [-0.575999943070835, 0.000262486079431672, -19.2299795908647, 0.00876275931760007, 0],
                            [-0.000262486080737868, -0.575999940191886, 0.00876276068239993, -19.2299765959399, 0],
                            [0, 0, 0, 0, 0]], dtype=float)
    p3.c_vector = np.array([0, 0, 0, 0, 1], dtype=float)
    p3.inv_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, 1.0], 125))  # t <= 125
    p3.inv_list.append(LinearConstraint([-1.0, 0.0, 0.0, 0.0, 0.0], 100))  # x >= -100
    p3.inv_list.append(LinearConstraint([1.0, 0.0, 0.0, 0.0, 0.0], 100))  # x <= 100
    p3.inv_list.append(LinearConstraint([0.0, -1.0, 0.0, 0.0, 0.0], 100))  # y >= -100
    p3.inv_list.append(LinearConstraint([0.0, 1.0, 0.0, 0.0, 0.0], 100))  # y <= 100
    p3.inv_list.append(LinearConstraint([-1.0, -1.0, 0.0, 0.0, 0.0], 141.1))  # x+y >= -141.1
    p3.inv_list.append(LinearConstraint([1.0, 1.0, 0.0, 0.0, 0.0], 141.1))  # x+y <= 141.1
    p3.inv_list.append(LinearConstraint([-1.0, 1.0, 0.0, 0.0, 0.0], 141.1))  # y-x <= 141.1
    p3.inv_list.append(LinearConstraint([1.0, -1.0, 0.0, 0.0, 0.0], 141.1))  # y-x >= -141.1

    passive = ha.new_mode('passive')
    passive.a_matrix = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0.0000575894721132000, 0, 0, 0.00876276, 0], [0, 0, -0.00876276, 0, 0],
                                 [0, 0, 0, 0, 0]], dtype=float)
    passive.c_vector = np.array([0, 0, 0, 0, 1], dtype=float)
    passive.inv_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, -1.0], -120))  # t >= 120

    trans1 = ha.new_transition(p2, p3)
    trans1.condition_list.append(LinearConstraint([-1.0, 0.0, 0.0, 0.0, 0.0], 100))  # x >= -100
    trans2 = ha.new_transition(p2, passive)
    trans2.condition_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, -1.0], -120))  # t >= 120
    trans3 = ha.new_transition(p3, passive)
    trans3.condition_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, -1.0], -120))  # t >= 120

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:
        #usafe_set_constraint_list.append(LinearConstraint([1.0, 0.0, 0.0, 0.0, 0.0], -100))  # x < -100
        usafe_set_constraint_list.append(LinearConstraint([1.0, 0.0, 0.0, 0.0, 0.0], -800))  # x < -800
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 0.0, 0.0, 0.0], 850))  # x > -850
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)
    trans4 = ha.new_transition(p2, error)
    trans5 = ha.new_transition(p3, error)
    trans6 = ha.new_transition(passive, error)
    for pred in usafe_set_constraint_list:
        trans4.condition_list.append(pred.clone())
        trans5.condition_list.append(pred.clone())
        trans6.condition_list.append(pred.clone())

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    rv = [(ha.modes['P2'], init_r)]

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    settings = HylaaSettings(step=0.1, max_time=10, plot_settings=plot_settings)
    settings.stop_when_error_reachable = False

    return settings


def run_hylaa(settings, init_r, usafe_r):
    '''run hylaa with the given settings, returning the HylaaResult object.'''
    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)

    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(-925, -875), (-425, -375), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])
    new_pv_object = run_hylaa(settings, init_r, None)
    new_pv_object.compute_longest_ce()
    #longest_ce = new_pv_object.compute_longest_ce()
    #depth_direction = np.identity(len(init_r.dims))
    #deepest_ce = new_pv_object.compute_deepest_ce(depth_direction[0])