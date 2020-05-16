'Counter-example trace generated using HyLAA'

import sys
import numpy as np
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    # dynamics x' = Ax + Bu + c
    a_matrix = np.array([\
        [-1.4142000000000001, 0.0, 0.0, 0.0], \
        [0.0, -1.4142000000000001, 0.0, 0.0], \
        [0.0, 0.0, -1.4142000000000001, 0.0], \
        [0.0, 0.0, 0.0, -1.4142000000000001], \
        ])

    b_matrix = None
    c_vector = np.array([0.0, 0.0, 0.0, 0.0])

    inputs = None
    end_point = [-1.8066109127540495e-06, -2.5292552778556692e-06, 1.0839665476524297e-06, -2.5292552778556692e-06]
    start_point = [-2.5, -3.5, 1.5, -3.5]

    step = 0.05
    max_time = 10.0

    normal_vec = [1.0, 0.0, 0.0, 0.0]
    normal_val = 1.0

    sim_states, sim_times = check(a_matrix, b_matrix, c_vector, step, max_time, start_point, inputs, end_point)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
