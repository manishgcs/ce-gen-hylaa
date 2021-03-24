from hylaa.starutil import DiscretePostParent
import subprocess
import re
from subprocess import Popen
import sys
sys.setrecursionlimit(2000)
from os import environ
from hylaa.timerutil import Timers
import numpy as np
from z3 import *


class CeSmt(object):
    def __init__(self, PV_object):
        self.pv_object = PV_object
        self.num_dims = PV_object.num_dims

    @staticmethod
    def convert_preds_into_z3_str(preds):
        z3_constraints = ''
        for pred in preds:
            p_into_list = pred.vector.tolist()
            if z3_constraints is not '':
                z3_constraints = z3_constraints + ', '
            z3_pred_str = ''
            for index in range(len(p_into_list)):
                if p_into_list[index] != 0:
                    if z3_pred_str is not '':
                        z3_pred_str = z3_pred_str + ' + '
                    z3_pred_str = z3_pred_str + str(p_into_list[index]) + '*x_' + str(index + 1)

            z3_pred_str = z3_pred_str + ' <= ' + str(pred.value)
            z3_constraints = z3_constraints + z3_pred_str
        return z3_constraints

    def transform_into_z3_problem(self, regex_string, init_node, path):
        prev_node_state = init_node.state
        with open("z3_predicates.py", 'w') as z3_preds_file:
            z3_preds_file.write('from z3 import *\n\n')
            for dim in range(self.num_dims):
                x_i = 'x_' + str(dim + 1)
                z3_preds_file.write('{} = Real(\'{}\')\n'.format(x_i, x_i))
            basis_centers = []

            z3_constraints = self.convert_preds_into_z3_str(init_node.state.constraint_list)
            z3_preds_file.write('init = And({})\n'.format(z3_constraints))
            z3_preds_file.write('s = Solver()\ns.add(init)\n')
            z3_preds_file.write('set_option(rational_to_decimal=True)\n')
            for str_index in range(len(regex_string)):
                character = regex_string[str_index]
                node = path[str_index]
                if prev_node_state.mode.name != node.state.mode.name:
                    if isinstance(node.state.parent.star.parent, DiscretePostParent):
                        basis_centers.append(node.state.parent.star.parent.prestar_basis_center)
                        prev_node_state = node.state
                usafe_basis_preds = self.pv_object.compute_usafe_set_pred_in_star_basis(node.state)
                for pred_index in range(len(usafe_basis_preds)):
                    pred = usafe_basis_preds[pred_index]
                    for basis_center in basis_centers[::-1]:
                        pred = self.pv_object.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                    usafe_basis_preds[pred_index] = pred

                z3_constraints = self.convert_preds_into_z3_str(usafe_basis_preds)
                z3_preds_file.write('b_{} = And({})\n'.format(str_index + 1, z3_constraints))
                if character is '1':
                    z3_preds_file.write('s.push()\n')
                    z3_preds_file.write('s.add(b_{})\n'.format(str_index + 1))
                elif character is '0':
                    z3_preds_file.write('s.push()\n')
                    z3_preds_file.write('s.add(Not(b_{}))\n'.format(str_index + 1))
            z3_preds_file.write('if s.check() == sat:\n')
            z3_preds_file.write('\tm = s.model()\n')
            z3_preds_file.write('\tprint(m)\n')
            z3_preds_file.write('else:\n')
            z3_preds_file.write('\tprint(\u0027No solution\u0027)')
        # pid = subprocess.Popen([sys.executable, "z3_predicates.py"])  # call subprocess
        env = dict(environ)
        args = ['python3', 'z3_predicates.py']
        p = Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        # p = subprocess.Popen('python z3_predicates.py', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout.readlines():
            print(line)
            if line != "No solution\n":
                print("Counter-example conforming the regular expression ({}) is: {}".format(regex_string, line))
                # return line.decode('UTF-8')
            else:
                print("Counter-example conforming the regular expression ({}) is: {}".format(regex_string,
                                                                                             "No solution"))
        z3_preds_file.close()
        return None

        # b_vars_list.append('b_'+str(node_index+1))

        # for b_index in xrange(len(b_vars_list)):
        #    z3_preds_file.write('s.push()\n')
        #    z3_preds_file.write('s.add({})\n'.format(b_vars_list[b_index]))
        # for b_index in xrange(len(b_vars_list)):
        #    z3_preds_file.write('if s.check() == sat:\n')
        #    z3_preds_file.write('\tm = s.model()\n')
        #    z3_preds_file.write('\tprint m\n')
        #    z3_preds_file.write('s.pop()\n')

    def compute_z3_ces_for_regex_file(self, str_length):
        counter_examples = []
        start_node = self.pv_object.reach_tree.nodes[0]
        current_path = []
        paths = []
        k_strs = []
        self.pv_object.explore_tree_for_error_stars(start_node, current_path, paths)
        path = paths[0]
        Timers.tic("Time taken for computing all string of given length")
        # k_strs = ["".join(seq) for seq in itertools.product("01", repeat=str_length)]
        Timers.toc("Time taken for computing all string of given length")

        k_strs.append("11111001111111111111111110")  # damped Osc 1
        # k_strs.append("11011111111100000")  # Ball String Medium
        # k_strs = ['111111111111111110']  # Ball String Medium
        # k_strs = ['00000111111111110']
        # k_strs.append("11111111011111111111111111111111111111111111111111111111111111111111111") # damped Osc 2
        # k_strs = ['0011111111111111111']  # Filtered Osc 32
        # k_strs = ['00000001111111111111111110']  # Damped Oscillator
        # k_strs = ['11111111111111111111111111111111111111111111111111111111111111111111111111']  # Vehicle Platoon 5
        # k_strs = ['11111111111111111111111111111111111111111111111111111111111111111']  # Vehicle Platoon 10
        # k_strs = ['00000011111111'] # Two tanks small
        # k_strs = ['00000000000000111111111111111111111111111111111111111111110']  # Two tanks medium
        # k_strs = ['111111111111111111110000000000000011111111111111111111111111111111111111111111111111']
        # Filtered Osc 4
        # k_strs = ['01111110000011111111111111111001']  # Buck Converter
        # k_strs = ['1111111111111111111111111111111']  # Ball String Large

        Timers.tic("Time taken by z3 to find counterexamples for all strings of given length")
        with open("z3_counterexamples", 'w') as z3_ces:
            for str in k_strs:
                current_ce = self.transform_into_z3_problem(str, start_node, path)
                counter_example = np.zeros(self.num_dims)
                if current_ce is not None:
                    z3_ces.write('{}'.format(current_ce))
                    current_ce_tokens = re.split(r'(=|,|\[|\])', current_ce)
                    print(current_ce_tokens)
                    for idx in range(len(current_ce_tokens)):
                        token = current_ce_tokens[idx]
                        if len(token) > 1:
                            if token[0] is 'x':
                                index = int(token[2:-1])
                                val = current_ce_tokens[idx + 2]
                                if val[len(val) - 1] is '?':
                                    val = val[:-1]
                                counter_example[index - 1] = float(val)
                                idx += 2
                            elif token[1] is 'x':
                                index = int(token[3:-1])
                                val = current_ce_tokens[idx + 2]
                                if val[len(val) - 1] is '?':
                                    val = val[:-1]
                                counter_example[index - 1] = float(val)
                                idx += 2
                    counter_examples.append(counter_example)
        z3_ces.close()
        Timers.toc("Time taken by z3 to find counterexamples for all strings of given length")
        return counter_examples

    def compute_z3_counterexample_file(self):
        start_node = self.pv_object.reach_tree.nodes[0]
        # prev_node_state = start_node.state
        current_path = []
        paths = []
        Timers.tic("Time taken by SMT")
        self.pv_object.explore_tree_for_error_stars(start_node, current_path, paths)
        path_id = 0
        print("No of paths: {}".format(len(paths)))
        for path in paths:
            prev_node_state = start_node.state
            print("No of nodes in the path is: '{}'".format(len(path)))
            file_name = "../preds_files/z3_preds_" + str(path_id)
            file_name += '.py'
            with open(file_name, 'w') as z3_preds_file:
                z3_preds_file.write('from z3 import *\n\n')
                for dim in range(self.num_dims):
                    x_i = 'x_' + str(dim + 1)
                    z3_preds_file.write('{} = Real(\'{}\')\n'.format(x_i, x_i))

                z3_preds_file.write('s = Optimize()\n')
                z3_preds_file.write('set_option(rational_to_decimal=True)\n')
                basis_centers = []
                for node_id in range(len(path)):

                    node = path[node_id]
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

                    z3_constraints = self.convert_preds_into_z3_str(usafe_basis_preds)
                    c_i = 'c_' + str(node_id + 1)
                    z3_preds_file.write('{} = Bool(\'{}\')\n'.format(c_i, c_i))
                    z3_preds_file.write('s.add({} == And({}))\n'.format(c_i, z3_constraints))
                    z3_preds_file.write('s.add_soft({})\n'.format(c_i))
                z3_preds_file.write('if s.check() == sat:\n')
                z3_preds_file.write('\tm = s.model()\n')
                z3_preds_file.write('\tprint(m)\n')
                z3_preds_file.write('else:\n')
                z3_preds_file.write('\tprint(\u0027No solution\u0027)')
            z3_preds_file.close()
            env = dict(environ)
            args = ['python3', file_name]
            p = Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            for line in p.stdout.readlines():
                print(line)
            z3_preds_file.close()
            path_id = path_id + 1
        Timers.toc("Time taken by SMT")

    def compute_counterexample(self, regex=None):
        if regex is None:
            self.compute_ce_wo_regex()
        else:
            self.compute_ce_w_regex(regex=regex)

    def compute_ce_wo_regex(self):
        start_node = self.pv_object.reach_tree.nodes[0]
        current_path = []
        paths = []
        Timers.tic("Time taken by SMT")
        self.pv_object.explore_tree_for_error_stars(start_node, current_path, paths)
        print("No of paths: {}".format(len(paths)))
        for path in paths:
            prev_node_state = start_node.state
            print("No of nodes in the path is: '{}'".format(len(path)))

            s = Optimize()
            set_option(rational_to_decimal=True)

            alpha = []
            for dim in range(self.num_dims):
                alpha_i = 'alpha_' + str(dim + 1)
                alpha.append(Real(alpha_i))

            basis_centers = []

            z = []
            n_z_vars = len(path)

            for node_id in range(len(path)):

                node = path[node_id]
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

                z_idx = 'z_' + str(node_id + 1)
                z.append(Bool(z_idx))

                c_idx = True
                for idy in range(len(usafe_basis_preds)):
                    pred_lhs = usafe_basis_preds[idy].vector.tolist()
                    pred_rhs = usafe_basis_preds[idy].value
                    c_idy = alpha[0] * pred_lhs[0]

                    for idz in range(1, self.num_dims):
                        c_idy = c_idy + alpha[idz] * pred_lhs[idz]

                    c_idx = And(c_idx, c_idy <= pred_rhs)

                s.add(z[node_id] == c_idx)
                s.add_soft(z[node_id])

            z_vals = []
            alpha_vals = []

            # opt_cost = Real('opt_cost')
            # for z_idx in range(0, n_z_vars):
            #     opt_cost = opt_cost + z[z_idx]
            if s.check() == sat:
                mdl = s.model()

                for idx in range(n_z_vars):
                    z_vals.append(mdl[z[idx]])

                for idx in range(self.num_dims):
                    alpha_vals.append(mdl[alpha[idx]])

                print(z_vals, alpha_vals)
        Timers.toc("Time taken by SMT")

    def compute_ce_w_regex(self, regex):
        start_node = self.pv_object.reach_tree.nodes[0]
        current_path = []
        paths = []
        self.pv_object.explore_tree_for_error_stars(start_node, current_path, paths)
        path = paths[0]
        # Timers.tic("Time taken for computing all string of given length")
        # k_strs = ["".join(seq) for seq in itertools.product("01", repeat=str_length)]
        # Timers.toc("Time taken for computing all string of given length")

        Timers.tic("Time taken by SMT")
        for regex_string in regex:

            init_node_constraints = start_node.state.constraint_list
            s = Solver()
            set_option(rational_to_decimal=True)

            alpha = []
            for dim in range(self.num_dims):
                alpha_i = 'alpha_' + str(dim + 1)
                alpha.append(Real(alpha_i))

            basis_centers = []
            init = True
            for idy in range(len(init_node_constraints)):
                pred_lhs = init_node_constraints[idy].vector.tolist()
                pred_rhs = init_node_constraints[idy].value
                c_idy = alpha[0] * pred_lhs[0]

                for idz in range(1, self.num_dims):
                    c_idy = c_idy + alpha[idz] * pred_lhs[idz]

                init = And(init, c_idy <= pred_rhs)

            s.add(init)

            prev_node_state = start_node.state

            z = []
            for str_index in range(len(regex_string)):
                z_idx = 'z_' + str(str_index + 1)
                z.append(Bool(z_idx))
                character = regex_string[str_index]
                if character is '1':
                    s.add(z[str_index])
                elif character is '0':
                    s.add(Not(z[str_index]))

            for str_index in range(len(regex_string)):
                node = path[str_index]
                if prev_node_state.mode.name != node.state.mode.name:
                    if isinstance(node.state.parent.star.parent, DiscretePostParent):
                        basis_centers.append(node.state.parent.star.parent.prestar_basis_center)
                        prev_node_state = node.state
                usafe_basis_preds = self.pv_object.compute_usafe_set_pred_in_star_basis(node.state)
                for pred_index in range(len(usafe_basis_preds)):
                    pred = usafe_basis_preds[pred_index]
                    for basis_center in basis_centers[::-1]:
                        pred = self.pv_object.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                    usafe_basis_preds[pred_index] = pred

                c_idx = True
                for idy in range(len(usafe_basis_preds)):
                    pred_lhs = usafe_basis_preds[idy].vector.tolist()
                    pred_rhs = usafe_basis_preds[idy].value
                    c_idy = alpha[0] * pred_lhs[0]

                    for idz in range(1, self.num_dims):
                        c_idy = c_idy + alpha[idz] * pred_lhs[idz]

                    c_idx = And(c_idx, c_idy <= pred_rhs)

                s.add(z[str_index] == c_idx)

            alpha_vals = []

            if s.check() == sat:
                mdl = s.model()

                for idx in range(self.num_dims):
                    alpha_vals.append(mdl[alpha[idx]])

            print(alpha_vals)
        Timers.toc("Time taken by SMT")