
import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, HyperRectangle, LinearConstraint
from hylaa.star import init_hr_to_star
from hylaa.engine import HylaaSettings, HylaaEngine
from hylaa.containers import PlotSettings
from hylaa.timerutil import Timers
from hylaa.pv_container import PVObject


def define_ha(settings, usafe_r):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["e1", "e1p", "a1", "e2", "e2p", "a2", "e3", "e3p", "a3", "e4", "e4p", "a4", "e5", "e5p", "a5"]

    loc1 = ha.new_mode('loc1')

    loc1.a_matrix = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.7152555329, 3.9705119979, -4.3600526739, -0.9999330812, -1.5731541104, 0.2669165553, -0.2215507198,
         -0.4303855023, 0.0669078193, -0.0881500219, -0.1881468451, 0.0322187056, -0.0343095071, -0.0767587194,
         0.0226660281],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.7153224517, 2.3973578876, 0.2669165553, 1.4937048131, 3.5401264957, -4.2931448546, -1.0880831031,
         -1.7613009555, 0.2991352608, -0.2558602268, -0.5071442217, 0.0895738474, -0.0881500219, - 0.1881468451,
         0.0548847337],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0.493771732, 1.9669723853, 0.0669078193, 0.6271724298, 2.2092110425, 0.2991352608, 1.4593953061, 3.4633677762,
         -4.2704788265, -1.0880831031, -1.7613009555, 0.3218012889, -0.2215507198, -0.4303855023, 0.121792553],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0],
        [0.40562171, 1.7788255402, 0.0322187056, 0.4594622249, 1.8902136659, 0.0895738474, 0.6271724298, 2.2092110425,
         0.3218012889, 1.4937048131, 3.5401264957, -4.2382601209, -0.9999330812, -1.5731541104, 0.3887091083],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1],
        [0.371312203, 1.7020668208, 0.0226660281, 0.40562171, 1.7788255402, 0.0548847337, 0.493771732, 1.9669723853,
         0.121792553, 0.7153224517, 2.3973578876, 0.3887091083, 1.7152555329, 3.9705119979, -3.9713435656],
        ], dtype=float)
    loc1.c_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

    # loc1.set_dynamics(a_matrix, c_vector)

    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(loc1, error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        # Create the list of constraints defining unsafe set if hyper-rectangle representation is not given
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0], 2))
        usafe_set_constraint_list.append(LinearConstraint([0.0, -1.0], -1))
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
    plot_settings.xdim = 0
    plot_settings.ydim = 1

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
    init_r = HyperRectangle([(0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1),
                             (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1)])

    # usafe_r = HyperRectangle(
    # [(-0.5, 0.91), (-0.5, 0.91), (-0.5, 0.92), (-0.5, 0.91), (-0.5, 0.92), (-0.5, 0.93), (-0.5, 0.9), (-0.5, 0.9),
    # (-0.5, 0.91), (-0.5, 0.93), (-0.5, 1.0), (-0.5, 0.9), (-0.5, 0.92), (-0.5, 0.93), (-0.5, 0.91)])

    # usafe_r = HyperRectangle(
    #   [(-0.5, 0.91), (-0.5, 0.91), (-0.5, 0.92), (-0.5, 0.91), (-0.5, 0.92), (-0.5, 0.93), (-0.5, 0.9), (-0.5, 0.9),
    #     (-0.5, 0.91), (-0.5, 0.93), (-0.5, -0.1), (-0.5, 0.9), (-0.5, 0.92), (-0.5, 0.93), (-0.5, -0.2)])

    # usafe_r = HyperRectangle(
    #   [(-0.5, 0.91), (-0.5, 0.91), (-0.5, 0.92), (-0.5, 0.91), (-0.5, 0.92), (-0.5, 0.93), (-0.5, 0.9), (-0.5, 0.9),
    #    (-0.5, 0.91), (-0.5, 0.93), (-0.5, 1.0), (-0.5, 0.9), (-0.5, 0.92), (-0.5, 0.93), (-0.5, -0.0)])

    usafe_r = HyperRectangle(
        [(-0.5, 0.91), (-0.5, 0.91), (-0.5, 0.92), (-0.5, 0.91), (-0.5, 0.92), (-0.5, 0.93), (-0.5, 0.91), (-0.5, 0.91),
        (-0.5, 0.91), (-0.5, 0.93), (-0.5, 1.0), (-0.5, 0.9), (-0.5, 0.92), (-0.5, 0.93), (-0.5, 0.2)])

    pv_object = run_hylaa(settings, init_r, usafe_r)
    # longest_ce = new_pv_object.compute_longest_ce()
    # depth_direction = np.identity(len(init_r.dims))
    # pv_object.compute_deepest_ce(depth_direction[2])

    # z3_counter_examples = pv_object.compute_counter_examples_using_z3(4)
    # pv_object.compute_milp_counterexamples()
    pv_object.compute_z3_counterexamples()
    Timers.print_stats()
