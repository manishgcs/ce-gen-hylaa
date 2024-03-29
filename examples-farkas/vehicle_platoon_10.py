import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint, HyperRectangle
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.plotutil import PlotSettings
from hylaa.star import init_hr_to_star
from hylaa.timerutil import Timers
from hylaa.pv_container import PVObject
from controlcore.control_utils import get_input, extend_a_b
from farkas_central.bdd4Ce import BDD4CE
from hylaa.ce_smt import CeSmt
from hylaa.ce_milp import CeMilp
from hylaa.simutil import compute_simulation
import matplotlib.pyplot as plt


def define_ha(settings, usafe_r=None):
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["e1", "e1p", "a1", "e2", "e2p", "a2", "e3", "e3p", "a3", "e4", "e4p", "a4", "e5", "e5p", "a5", "e6", "e6p", "a6", "e7", "e7p", "a7", "e8", "e8p", "a8", "e9", "e9p", "a9", "e10", "e10p", "a10"]

    loc1 = ha.new_mode('loc1')

    loc1.a_matrix = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.702423734, 3.929356551, -4.3607983776, -1.01374489, -1.6167727749, 0.2653009364, -0.2375199245,
         -0.4793543458, 0.06412815, -0.1079326841, -0.2463610381, 0.0276872161, -0.0605561959, -0.1501445039,
         0.0151944922, -0.0374830081, -0.0986391305, 0.009628751, -0.0242136837, -0.0665592904, 0.0067836913,
         -0.015601062, -0.0442510048, 0.0052325207, 0.0093924696, 0.0272127915, 0.0043984935, -0.0044278796,
         -0.0129879863, 0.0040303349],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.688678844, 2.3125837761, 0.2653009364, 1.4649038095, 3.4500022052, -4.2966702275, -1.1216775741,
         -1.863133813, 0.2929881525, -0.2980761204, -0.6294988497, 0.0793226422, -0.1454156921, -0.3450001686,
         0.0373159671, -0.0847698796, -0.2167037943, 0.0219781835, -0.0530840701, -0.1428901352, 0.0148612718,
         -0.0336061533, -0.0937720819, 0.0111821848, -0.0200289416, -0.057238991, 0.0092628557, -0.0093924696,
         -0.0272127915, 0.0084288284],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.4511589195, 1.8332294303, 0.06412815, 0.5807461599, 2.066222738, 0.2929881525, 1.4043476136, 3.2998577013,
         -4.2814757354, -1.1591605822, -1.9617729435, 0.3026169036, -0.3222898041, -0.6960581401, 0.0861063336,
         -0.1610167541, -0.3892511733, 0.0425484878, -0.0941623492, -0.2439165858, 0.026376677, -0.0575119497,
         -0.1558781215, 0.0188916067, -0.0336061533, -0.0937720819, 0.0152125197, -0.015601062, -0.0442510048,
         0.0136613491],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.3432262354, 1.5868683922, 0.0276872161, 0.3906027236, 1.6830849264, 0.0793226422, 0.5432631518, 1.9675836075,
         0.3026169036, 1.3801339299, 3.2332984109, -4.274692044, -1.1747616442, -2.0060239482, 0.3078494243,
         -0.3316822737, -0.7232709316, 0.090504827, -0.1654446337, -0.4022391596, 0.0465788228, -0.0941623492,
         -0.2439165858, 0.0304070119, -0.0530840701, -0.1428901352, 0.0232901001, -0.0242136837, -0.0665592904,
         0.0204450405],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.2826700395, 1.4367238883, 0.0151944922, 0.3057432273, 1.4882292617, 0.0373159671, 0.3663890398, 1.616525636,
         0.0861063336, 0.5276620899, 1.9233326028, 0.3078494243, 1.3707414603, 3.2060856194, -4.2702935506,
         -1.1791895238, -2.0190119345, 0.3118797592, -0.3316822737, -0.7232709316, 0.094535162, -0.1610167541,
         -0.3892511733, 0.0509773162, -0.0847698796, -0.2167037943, 0.0356395326, -0.0374830081, -0.0986391305,
         0.0300737915],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.2451870315, 1.3380847578, 0.009628751, 0.2584563558, 1.3701645979, 0.0219781835, 0.2901421653, 1.443978257,
         0.0425484878, 0.3569965702, 1.5893128445, 0.090504827, 0.5232342102, 1.9103446165, 0.3118797592, 1.3707414603,
         3.2060856194, -4.2662632156, -1.1747616442, -2.0060239482, 0.3162782527, -0.3222898041, -0.6960581401,
         0.0997676827, -0.1454156921, -0.3450001686, 0.0577610076, -0.0605561959, -0.1501445039, 0.0452682837],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.2209733477, 1.2715254674, 0.0067836913, 0.2295859695, 1.2938337531, 0.0148612718, 0.2490638862, 1.3429518064,
         0.026376677, 0.2857142857, 1.4309902707, 0.0465788228, 0.3569965702, 1.5893128445, 0.094535162, 0.5276620899,
         1.9233326028, 0.3162782527, 1.3801339299, 3.2332984109, -4.2610306949, -1.1591605822, -1.9617729435,
         0.323061944, -0.2980761204, -0.6294988497, 0.1093964337, 0.1079326841, -0.2463610381, 0.0729554998],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0.2053722857, 1.2272744627, 0.0052325207, 0.2115808781, 1.2443126759, 0.0111821848, 0.2251580898, 1.2808457668,
         0.0188916067, 0.2490638862, 1.3429518064, 0.0304070119, 0.2901421653, 1.443978257, 0.0509773162, 0.3663890398,
         1.616525636, 0.0997676827, 0.5432631518, 1.9675836075, 0.323061944, 1.4043476136, 3.2998577013, -4.2514019439,
         -1.1216775741, -1.863133813, 0.3382564362, -0.2375199245, -0.4793543458, 0.1370836498],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0],
        [0.1959798161, 1.2000616712, 0.0043984935, 0.2009444061, 1.2142864764, 0.0092628557, 0.2115808781, 1.2443126759,
         0.0152125197, 0.2295859695, 1.2938337531, 0.0232901001, 0.2584563558, 1.3701645979, 0.0356395326, 0.3057432273,
         1.4882292617, 0.0577610076, 0.3906027236, 1.6830849264, 0.1093964337, 0.5807461599, 2.066222738, 0.3382564362,
         1.4649038095, 3.4500022052, -4.2237147278, -1.01374489, -1.6167727749, 0.4023845862],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1],
        [0.1915519365, 1.1870736849, 0.0040303349, 0.1959798161, 1.2000616712, 0.0084288284, 0.2053722857, 1.2272744627,
         0.0136613491, 0.2209733477, 1.2715254674, 0.0204450405, 0.2451870315, 1.3380847578, 0.0300737915, 0.2826700395,
         1.4367238883, 0.0452682837, 0.3432262354, 1.5868683922, 0.0729554998, 0.4511589195, 1.8332294303, 0.1370836498,
         0.688678844, 2.3125837761, 0.4023845862, 1.702423734, 3.929356551, -3.9584137913]
        ], dtype=float)
    loc1.c_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             dtype=float)

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:

        # exp 1
        # usafe_set_constraint_list.append(LinearConstraint([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], -0.37))
        usafe_set_constraint_list.append(
            LinearConstraint([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             -2.5))
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
    # Variable ordering: [x, v]
    rv = [(ha.modes['loc1'], init_r)]

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    settings = HylaaSettings(step=0.2, max_time=20.0, disc_dyn=False, plot_settings=plot_settings)
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

    init_r = HyperRectangle([(0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1),
                             (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1),
                             (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1),
                             (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (0.9, 1.1),
                             (0.9, 1.1), (0.9, 1.1)])

    pv_object = run_hylaa(settings, init_r, None)

    # longest_ce = pv_object.compute_longest_ce()

    # mid-order = +2
    # random: [5, 3, 0, 7, 8, 9, 6, 1, 4, 2]
    bdd_ce_object = BDD4CE(pv_object, equ_run=False, smt_mip='mip')
    bdd_graphs = bdd_ce_object.create_bdd_w_level_merge(level_merge=0, order='mid-order')
    valid_exps, invalid_exps = bdd_graphs[0].generate_expressions()
    print(len(valid_exps), len(invalid_exps))

    # [5, 3, 0, 7, 8, 9, 6, 1, 4, 2]

    Timers.print_stats()

