'Counter-example trace generated using HyLAA'

import sys
import numpy as np
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    # dynamics x' = Ax + Bu + c
    a_matrix = np.array([\
        [-23.2999, -2.2873, 19.8882, -0.0153], \
        [0.0, 1.0, 0.0, 0.1], \
        [-24.2999, -2.2873, 20.6752, -0.0063999999999999994], \
        [-0.0771, -46.7949, 0.0707, -3.3383], \
        ])

    b_matrix = None
    c_vector = np.array([0.0, 0.0, 0.0, 0.0])

    inputs = None
    end_point = [-0.0027364944874395006, 0.0004925861137463548, -0.003033139295727563, -0.009843865712222738]
    start_point = [0.7, 1.7, 0.0, 0.0]

    step = 0.02
    max_time = 10.0

    normal_vec = [1.0, 0.0, 0.0, 0.0]
    normal_val = 1.0

    sim_states, sim_times = check(a_matrix, b_matrix, c_vector, step, max_time, start_point, inputs, end_point)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
