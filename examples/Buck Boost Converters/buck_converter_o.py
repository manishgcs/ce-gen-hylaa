
import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint, HyperRectangle
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.plotutil import PlotSettings
from hylaa.star import init_hr_to_star
from hylaa.pv_container import PVObject
from hylaa.simutil import compute_simulation
import matplotlib.pyplot as plt


def define_ha(settings, usafe_r=None):
    '''make the hybrid automaton and return it'''

    # cap_d = 0.41667
    # vs = 12
    # t_max = 0.00025
    cap_d = 0.42857
    cap_t = 0.00005
    vs = 20
    t_max = 0.001

    ha = LinearHybridAutomaton()
    ha.variables = ["il", "vc", "t", "gt"]

    open1 = ha.new_mode('open1')
    open1.a_matrix = np.array([[0, -21052.6316, 0, 0], [42105.2632, -40100.2506, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              dtype=float)
    open1.c_vector = np.array([0 * vs, 0 * vs, 1, 1], dtype=float)
    open1.inv_list.append(LinearConstraint([0.0, 0.0, -1, 0.0], -0))  # t >= 0
    open1.inv_list.append(LinearConstraint([0.0, 0.0, 1, 0.0], cap_d * cap_t))  # t <= DT
    open1.inv_list.append(LinearConstraint([0.0, 0.0, 0, -1], -0))  # gt >= 0
    open1.inv_list.append(LinearConstraint([0.0, 0.0, 0, 1], t_max))  # gt <= tmax
    open1.inv_list.append(LinearConstraint([-1, 0.0, 0.0, 0.0], 1000))  # il >= -bounds
    open1.inv_list.append(LinearConstraint([1, 0.0, 0.0, 0.0], 1000))  # il <= bounds
    open1.inv_list.append(LinearConstraint([0.0, -1, 0, 0.0], 1000))  # vc >= -bounds
    open1.inv_list.append(LinearConstraint([0.0, 1, 0, 0.0], 1000))  # vc <= bounds

    closed1 = ha.new_mode('closed1')
    closed1.a_matrix = np.array([[0, -21052.6316, 0, 0], [42105.2632, -40100.2506, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                dtype=float)
    closed1.c_vector = np.array([21052.6316 * vs, 0 * vs, 1, 1], dtype=float)
    closed1.inv_list.append(LinearConstraint([0.0, 0.0, -1, 0.0], -(cap_d * cap_t)))  # t >= DT
    closed1.inv_list.append(LinearConstraint([0.0, 0.0, 1, 0.0], cap_t))  # t <= T
    closed1.inv_list.append(LinearConstraint([0.0, 0.0, 0, -1], -0))  # gt >= 0
    closed1.inv_list.append(LinearConstraint([0.0, 0.0, 0, 1], t_max))  # gt <= tmax
    closed1.inv_list.append(LinearConstraint([-1, 0.0, 0.0, 0.0], 1000))  # il >= -bounds
    closed1.inv_list.append(LinearConstraint([1, 0.0, 0.0, 0.0], 1000))  # il <= bounds
    closed1.inv_list.append(LinearConstraint([0.0, -1, 0, 0.0], 1000))  # vc >= -bounds
    closed1.inv_list.append(LinearConstraint([0.0, 1, 0, 0.0], 1000))  # vc <= bounds

    open2 = ha.new_mode('open2')
    open2.a_matrix = np.array([[0, -21052.6316, 0, 0], [42105.2632, -40100.2506, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              dtype=float)
    open2.c_vector = np.array([0 * vs, 0 * vs, 1, 1], dtype=float)
    open2.inv_list.append(LinearConstraint([0.0, 0.0, -1, 0.0], -cap_t))  # t >= T
    open2.inv_list.append(LinearConstraint([0.0, 0.0, 1, 0.0], (1+cap_d)*cap_t))  # t <= T+DT
    open2.inv_list.append(LinearConstraint([0.0, 0.0, 0, -1], -0))  # gt >= 0
    open2.inv_list.append(LinearConstraint([0.0, 0.0, 0, 1], t_max))  # gt <= tmax
    open2.inv_list.append(LinearConstraint([-1, 0.0, 0.0, 0.0], 1000))  # il >= -bounds
    open2.inv_list.append(LinearConstraint([1, 0.0, 0.0, 0.0], 1000))  # il <= bounds
    open2.inv_list.append(LinearConstraint([0.0, -1, 0, 0.0], 1000))  # vc >= -bounds
    open2.inv_list.append(LinearConstraint([0.0, 1, 0, 0.0], 1000))  # vc <= bounds

    closed2 = ha.new_mode('closed2')
    closed2.a_matrix = np.array([[0, -21052.6316, 0, 0], [42105.2632, -40100.2506, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                dtype=float)
    closed2.c_vector = np.array([21052.6316 * vs, 0 * vs, 1, 1], dtype=float)
    closed2.inv_list.append(LinearConstraint([0.0, 0.0, -1, 0.0], -((1+cap_d)*cap_t)))  # t >= T+DT
    closed2.inv_list.append(LinearConstraint([0.0, 0.0, 1, 0.0], (2 * cap_t)))  # t <= 2T
    closed2.inv_list.append(LinearConstraint([0.0, 0.0, 0, -1], -0))  # gt >= 0
    closed2.inv_list.append(LinearConstraint([0.0, 0.0, 0, 1], t_max))  # gt <= tmax
    closed2.inv_list.append(LinearConstraint([-1, 0.0, 0.0, 0.0], 1000))  # il >= -bounds
    closed2.inv_list.append(LinearConstraint([1, 0.0, 0.0, 0.0], 1000))  # il <= bounds
    closed2.inv_list.append(LinearConstraint([0.0, -1, 0, 0.0], 1000))  # vc >= -bounds
    closed2.inv_list.append(LinearConstraint([0.0, 1, 0, 0.0], 1000))  # vc <= bounds

    open3 = ha.new_mode('open3')
    open3.a_matrix = np.array([[0, -21052.6316, 0, 0], [42105.2632, -40100.2506, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              dtype=float)
    open3.c_vector = np.array([0 * vs, 0 * vs, 1, 1], dtype=float)
    open3.inv_list.append(LinearConstraint([0.0, 0.0, -1, 0.0], -(2*cap_t)))  # t >= 2T
    open3.inv_list.append(LinearConstraint([0.0, 0.0, 1, 0.0], (2+cap_d)*cap_t))  # t <= 2T+DT
    open3.inv_list.append(LinearConstraint([0.0, 0.0, 0, -1], -0))  # gt >= 0
    open3.inv_list.append(LinearConstraint([0.0, 0.0, 0, 1], t_max))  # gt <= tmax
    open3.inv_list.append(LinearConstraint([-1, 0.0, 0.0, 0.0], 1000))  # il >= -bounds
    open3.inv_list.append(LinearConstraint([1, 0.0, 0.0, 0.0], 1000))  # il <= bounds
    open3.inv_list.append(LinearConstraint([0.0, -1, 0, 0.0], 1000))  # vc >= -bounds
    open3.inv_list.append(LinearConstraint([0.0, 1, 0, 0.0], 1000))  # vc <= bounds

    closed3 = ha.new_mode('closed3')
    closed3.a_matrix = np.array([[0, -21052.6316, 0, 0], [42105.2632, -40100.2506, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                dtype=float)
    closed3.c_vector = np.array([21052.6316 * vs, 0 * vs, 1, 1], dtype=float)
    closed3.inv_list.append(LinearConstraint([0.0, 0.0, -1, 0.0], -((2+cap_d)*cap_t)))  # t >= 2T+DT
    closed3.inv_list.append(LinearConstraint([0.0, 0.0, 1, 0.0], (3 * cap_t)))  # t <= 3T
    closed3.inv_list.append(LinearConstraint([0.0, 0.0, 0, -1], -0))  # gt >= 0
    closed3.inv_list.append(LinearConstraint([0.0, 0.0, 0, 1], t_max))  # gt <= tmax
    closed3.inv_list.append(LinearConstraint([-1, 0.0, 0.0, 0.0], 1000))  # il >= -bounds
    closed3.inv_list.append(LinearConstraint([1, 0.0, 0.0, 0.0], 1000))  # il <= bounds
    closed3.inv_list.append(LinearConstraint([0.0, -1, 0, 0.0], 1000))  # vc >= -bounds
    closed3.inv_list.append(LinearConstraint([0.0, 1, 0, 0.0], 1000))  # vc <= bounds

    trans1 = ha.new_transition(open1, closed1)
    trans1.condition_list.append(LinearConstraint([0.0, 0.0, -1, 0.0], -cap_d * cap_t))  # t >= DT
    trans2 = ha.new_transition(closed1, open2)
    trans2.condition_list.append(LinearConstraint([0.0, 0.0, -1, 0.0], -cap_t))  # t >= T
    trans3 = ha.new_transition(open2, closed2)
    trans3.condition_list.append(LinearConstraint([0.0, 0.0, -1, 0.0], -(1+cap_d) * cap_t))  # t >= T+DT
    trans4 = ha.new_transition(closed2, open3)
    trans4.condition_list.append(LinearConstraint([0.0, 0.0, -1, 0.0], -2*cap_t))  # t >= 2T
    trans5 = ha.new_transition(open3, closed3)
    trans5.condition_list.append(LinearConstraint([0.0, 0.0, -1, 0.0], -(2 + cap_d) * cap_t))  # t >= 2T+DT

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([0.0, -1.0, 0.0, 0.0], -4.05))
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)
    trans6 = ha.new_transition(open1, error)
    trans7 = ha.new_transition(closed1, error)
    trans8 = ha.new_transition(open2, error)
    for pred in usafe_set_constraint_list:
        trans6.condition_list.append(pred.clone())
        trans7.condition_list.append(pred.clone())
        trans8.condition_list.append(pred.clone())

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    rv = [(ha.modes['open1'], init_r)]

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 1
    plot_settings.ydim = 0

    settings = HylaaSettings(step=0.000001, max_time=0.00005, plot_settings=plot_settings)
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
    init_r = HyperRectangle([(0.0, 1), (0.0, 1), (0.0, 0.0), (0.0, 0.0)])
    pv_object = run_hylaa(settings, init_r, None)
    longest_ce = pv_object.compute_longest_ce()
    depth_direction = np.identity(len(init_r.dims))
    deepest_ce = pv_object.compute_deepest_ce(depth_direction[1])
    robust_ce = pv_object.compute_robust_ce()