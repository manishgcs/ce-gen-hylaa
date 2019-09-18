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
    ha.variables = ["temp", "t"]

    power = 7.0
    high = 22.0
    low = 18.0
    c = 0.4
    Tenv = 10.0

    on = ha.new_mode('on')
    on.a_matrix = np.array([[-c, 0.0], [0.0, 0.0]], dtype=float)
    on.c_vector = np.array([Tenv*c+power, 1.0], dtype=float)
    on.inv_list.append(LinearConstraint([1.0, 0.0], high))  # temp <= high

    off = ha.new_mode('off')
    off.a_matrix = np.array([[-c, 0.0], [0.0, 0.0]], dtype=float)
    off.c_vector = np.array([Tenv * c, 1.0], dtype=float)
    off.inv_list.append(LinearConstraint([-1.0, 0.0], -low))  # temp >= low

    trans1_2 = ha.new_transition(on, off)
    trans1_2.condition_list.append(LinearConstraint([-1.0, 0.0], -high))  # temp > high

    trans2_1 = ha.new_transition(off, on)
    trans2_1.condition_list.append(LinearConstraint([1.0, 0.0], low))  # temp < low

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0], -21))  # temp >= high
        # usafe_set_constraint_list.append(LinearConstraint([1.0, 0.0], low))  # temp <= low
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    trans1_error = ha.new_transition(on, error)
    trans2_error = ha.new_transition(off, error)
    for constraint in usafe_set_constraint_list:
        trans1_error.condition_list.append(constraint)
        trans2_error.condition_list.append(constraint)

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, v]
    rv = []

    rv.append((ha.modes['on'], init_r))

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 1
    plot_settings.ydim = 0

    settings = HylaaSettings(step=0.04, max_time=2.0, plot_settings=plot_settings)
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
    init_r = HyperRectangle([(20.0, 20.5), (0.0, 0.0)])

    pv_object = run_hylaa(settings, init_r, None)
    # new_pv_object.compute_milp_counterexamples()
    pv_object.compute_z3_counterexamples()
    Timers.print_stats()
