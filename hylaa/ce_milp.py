from hylaa.starutil import DiscretePostParent
import subprocess
import re
from subprocess import Popen
import sys
sys.setrecursionlimit(2000)
from os import environ
from hylaa.timerutil import Timers
import numpy as np
from hylaa.reach_regex_milp import ReachabilityInstance, RegexInstance
from hylaa.polytope import Polytope


class CeMilp(object):
    def __init__(self, PV_object):
        self.pv_object = PV_object
        self.num_dims = PV_object.num_dims

    def compute_counterexample(self, benchmark, regex=None):
        start_node = self.pv_object.reach_tree.nodes[0]
        current_path = []
        paths = []
        Timers.tic("Time taken by MILP")
        self.pv_object.explore_tree_for_error_stars(start_node, current_path, paths)
        for path in paths:

            prev_node_state = start_node.state
            print("No of nodes in the path is: '{}'".format(len(path)))

            milp_instance = ReachabilityInstance(benchmark, self.num_dims, len(path))
            if regex is not None:
                milp_instance = RegexInstance(benchmark, self.num_dims, len(path))

            basis_centers = []
            for node in path:

                if prev_node_state.mode.name != node.state.mode.name:
                    if isinstance(node.state.parent.star.parent, DiscretePostParent):
                        basis_centers.append(node.state.parent.star.parent.prestar_basis_center)
                        prev_node_state = node.state
                usafe_basis_preds = self.pv_object.compute_usafe_set_pred_in_star_basis(node.state)
                for index in range(len(usafe_basis_preds)):
                    pred = usafe_basis_preds[index]
                    for basis_center in basis_centers[::-1]:
                        pred = self.pv_object.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                    usafe_basis_preds[index] = pred

                no_of_constraints = len(usafe_basis_preds)
                con_matrix = np.zeros((no_of_constraints, self.num_dims), dtype=float)
                rhs = np.zeros(no_of_constraints, dtype=float)
                for idx in range(len(usafe_basis_preds)):
                    pred = usafe_basis_preds[idx]
                    pred_list = pred.vector.tolist()
                    for idy in range(len(pred_list)):
                        if pred_list[idy] != 0:
                            con_matrix[idx][idy] = pred_list[idy]
                    rhs[idx] = pred.value
                # print(con_matrix)
                polytope = Polytope(no_of_constraints, con_matrix, rhs)

                milp_instance.addPolytope(polytope)

            if regex is None:
                milp_instance.solve()
            else:
                milp_instance.solve(regex)

        Timers.toc("Time taken by MILP")

    def compute_milp_counterexample_cpp(self, benchmark):
        start_node = self.pv_object.reach_tree.nodes[0]
        current_path = []
        paths = []
        Timers.tic("Time taken by MILP")
        self.pv_object.explore_tree_for_error_stars(start_node, current_path, paths)
        path_id = 0
        for path in paths:
            prev_node_state = start_node.state
            print("No of nodes in the path is: '{}'".format(len(path)))
            file_name = "../preds_files/milp_preds_" + str(path_id)
            with open(file_name, 'w') as lin_preds_file:

                lin_preds_file.write('{}\n'.format(self.num_dims))
                lin_preds_file.write('{}\n'.format(len(path)))
                basis_centers = []
                for node in path:

                    if prev_node_state.mode.name != node.state.mode.name:
                        if isinstance(node.state.parent.star.parent, DiscretePostParent):
                            basis_centers.append(node.state.parent.star.parent.prestar_basis_center)
                            prev_node_state = node.state
                    usafe_basis_preds = self.pv_object.compute_usafe_set_pred_in_star_basis(node.state)
                    for index in range(len(usafe_basis_preds)):
                        pred = usafe_basis_preds[index]
                        for basis_center in basis_centers[::-1]:
                            pred = self.pv_object.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                        usafe_basis_preds[index] = pred

                    no_of_constraints = len(usafe_basis_preds)
                    lin_preds_file.write('{}\n'.format(no_of_constraints))
                    for pred in usafe_basis_preds:
                        pred_list = pred.vector.tolist()
                        pred_str = ''
                        no_of_vars_in_pred = 0
                        for index in range(len(pred_list)):
                            if pred_list[index] != 0:
                                no_of_vars_in_pred = no_of_vars_in_pred + 1
                                pred_str = pred_str + str(index+1) + ' ' + str(pred_list[index]) + ' '
                        pred_str = pred_str + str(pred.value)
                        lin_preds_file.write('{} {}\n'.format(no_of_vars_in_pred, pred_str))

            env = dict(environ)
            # /home/manish/gurobi810/linux64/examples/build
            # g++ -m64 -g -o ../c++/reach_regx/reach  ../c++/reach_regx/instance.cpp ../c++/reach_regx/main.cpp
            # -I../../include/ -I../c++/reach_regx -L../../lib/ -lgurobi_c++ -lgurobi81 -lm
            env['LD_LIBRARY_PATH'] = '/home/manishg/Research/gurobi810/linux64/lib'
            args = ['/home/manishg/Research/gurobi903/linux64/examples/c++/reach_opt/reach', benchmark, file_name]
            p = Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            for line in p.stdout.readlines():
                # if line.find("Bounds:") != -1:
                print(line)
            lin_preds_file.close()
            path_id = path_id + 1
        Timers.toc("Time taken by MILP")
