'''Automated verification of continuous and hybrid dynamical systems from William Denman'''

import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, HyperRectangle, LinearConstraint
from hylaa.engine import HylaaSettings, HylaaEngine
from hylaa.containers import PlotSettings
from hylaa.pv_container import PVObject
from hylaa.simutil import compute_simulation
from hylaa.timerutil import Timers
from hylaa.star import init_hr_to_star


def define_ha(settings, usafe_r):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2", "x3"]

    loc1 = ha.new_mode('loc1')

    loc1.a_matrix = np.array([[-0.1, -1, 0], [1, -0.1, 0], [0, 0, -0.15]])
    loc1.c_vector = np.array([0, 0, 0], dtype=float)
    error = ha.new_mode('_error')
    error.is_error = True

    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(loc1, error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        # usafe_set_constraint_list.append(LinearConstraint([1, 0, 0], 0.50))
        # usafe_set_constraint_list.append(LinearConstraint([-1, 0, 0], -0.25))
        usafe_set_constraint_list.append(LinearConstraint([1, 0, 0], 0.8))
        usafe_set_constraint_list.append(LinearConstraint([-1, 0, 0], -0.2))
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    for constraint in usafe_set_constraint_list:
        trans.condition_list.append(constraint)

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, y]
    rv = []
    rv.append((ha.modes['loc1'], init_r))

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_MATLAB
    plot_settings.xdim = 0
    plot_settings.ydim = 2

    s = HylaaSettings(step=0.2, max_time=20.0, plot_settings=plot_settings)
    s.stop_when_error_reachable = False

    return s


def run_hylaa(settings, init_r, usafe_r):
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)
    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


def compute_simulation_mt(smt_ce, milp_ce = None):

    a_matrix = np.array([[-0.1, -1, 0], [1, -0.1, 0], [0, 0, -0.15]])
    c_vector = np.array([0, 0, 0], dtype=float)
    smt_ce_simulation = compute_simulation(smt_ce, a_matrix, c_vector, 0.1, 200)
    milp_ce_simulation = compute_simulation(milp_ce, a_matrix, c_vector, 0.1, 200)

    with open("simulation", 'w') as f:
        f.write(' SMT_simulation = [')
        t = 0.0
        for point in smt_ce_simulation:
            f.write('{},{};\n'.format(str(point[0]), str(point[2])))
            t = t + 0.1
        f.write('];')
        f.write('\n**************************************\n')
        f.write('MILP_simulation = [')
        t = 0.0
        for point in milp_ce_simulation:
           f.write('{},{};\n'.format(str(point[0]), str(point[2])))
           t = t + 0.1
        f.write('];')


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(-0.1, 0.1), (0.8, 1), (0.9, 1)])

    # usafe_r = HyperRectangle([(0, 0.1), (0, 0.1), (0.5, 0.8)])

    pv_object = run_hylaa(settings, init_r, None)

    pv_object.compute_z3_counterexamples()
    pv_object.compute_milp_counterexamples('Particle')

    # milp_ce = np.array([-0.0932466, 0.819339, 0.9])
    # smt_ce = np.array([0.0384758974, 0.8877238855, 0.9])
    smt_ce = np.array([-0.033405184, 0.9732033863, 0.9])
    milp_ce = np.array([-0.0465493, 0.986873, 0.9])
    compute_simulation_mt(smt_ce, milp_ce)
    Timers.print_stats()