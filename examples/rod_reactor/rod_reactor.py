
import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint, HyperRectangle
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.plotutil import PlotSettings
from hylaa.star import init_hr_to_star
from hylaa.pv_container import PVObject
from hylaa.simutil import compute_simulation


def define_ha(settings, usafe_r=None):
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x", "c1", "c2"]

    norod = ha.new_mode('norod')
    norod.a_matrix = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    norod.c_vector = np.array([-50, 1.0, 1.0], dtype=float)
    norod.inv_list.append(LinearConstraint([1.0, 0.0, 0.0], 550))

    rod1 = ha.new_mode('rod1')
    rod1.a_matrix = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    rod1.c_vector = np.array([-56, 1.0, 1.0], dtype=float)
    rod1.inv_list.append(LinearConstraint([-1.0, 0.0, 0.0], -510))

    rod2 = ha.new_mode('rod2')
    rod2.a_matrix = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    rod2.c_vector = np.array([-60, 1.0, 1.0], dtype=float)
    rod2.inv_list.append(LinearConstraint([-1.0, 0.0, 0.0], -510))

    error = ha.new_mode('_error')
    error.is_error = True

    trans1 = ha.new_transition(norod, rod1)
    trans1.condition_list.append(LinearConstraint([-1.0, -0.0, -0.0], -550.0))
    trans1.condition_list.append(LinearConstraint([1.0, 0.0, 0.0], 550.0))
    trans1.condition_list.append(LinearConstraint([-0.0, -1.0, -0.0], -20.0))

    trans2 = ha.new_transition(norod, rod2)
    trans2.condition_list.append(LinearConstraint([-1.0, -0.0, -0.0], -550.0))
    trans2.condition_list.append(LinearConstraint([1.0, 0.0, 0.0], 550.0))
    trans2.condition_list.append(LinearConstraint([-0.0, -0.0, -1.0], -20.0))

    trans3 = ha.new_transition(rod1, norod)
    trans3.condition_list.append(LinearConstraint([-1.0, -0.0, -0.0], -510.0))
    trans3.condition_list.append(LinearConstraint([1.0, 0.0, 0.0], 510.0))
    trans3.reset_matrix = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    trans3.reset_vector = np.array([0.0, 0.0, 0.0], dtype=float)

    trans4 = ha.new_transition(rod2, norod)
    trans4.condition_list.append(LinearConstraint([-1.0, -0.0, -0.0], -510.0))
    trans4.condition_list.append(LinearConstraint([1.0, 0.0, 0.0], 510.0))
    trans4.reset_matrix = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    trans4.reset_vector = np.array([0.0, 0.0, 0.0], dtype=float)

    trans5 = ha.new_transition(norod, error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([-1.0, -0.0, -0.0], -550.0))
        usafe_set_constraint_list.append(LinearConstraint([1.0, 0.0, 0.0], 550.0))
        usafe_set_constraint_list.append(LinearConstraint([0.0, 1.0, 0.0], 20))
        usafe_set_constraint_list.append(LinearConstraint([0.0, 0.0, 1.0], 20))
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    for constraint in usafe_set_constraint_list:
        trans5.condition_list.append(constraint)

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, v]
    rv = []

    rv.append((ha.modes['norod'], init_r))

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    settings = HylaaSettings(step=0.02, max_time=2.0, plot_settings=plot_settings)
    settings.stop_when_error_reachable = False

    return settings


def run_hylaa(settings, init_r, usafe_r):
    'Runs hylaa with the given settings, returning the HylaaResult object.'
    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)

    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(540, 541), (10, 20), (10, 20)])

    # Direction(s) in which we intend to minimize co-efficients (alpha's)
    direction = np.ones(len(init_r.dims))

    # direction = np.identity(len(init_r.dims))
    pv_object = run_hylaa(settings, init_r, None)
    longest_ce = pv_object.compute_longest_ce()
