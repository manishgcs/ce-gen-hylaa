
' Post-verification container for reachTree implementation'

from os import environ
import numpy as np
from hylaa.glpk_interface import LpInstance
from hylaa.hybrid_automaton import LinearConstraint
from hylaa.starutil import InitParent, DiscretePostParent
from hylaa.timerutil import Timers
import subprocess
import itertools
import re
import time
from itertools import combinations
from numpy.linalg import inv
from subprocess import Popen


class CeObject(object):
    def __init__(self, ce, ce_length, ce_depth, patch, start_index, end_index):
        self.counterexample = ce
        self.ce_length = ce_length
        self.ce_depth = ce_depth
        self.patch = patch
        self.start_index = start_index
        self.end_index = end_index


class PVObject(object):
    def __init__(self, num_dims, usafe_set_constraint_list, reach_tree):

        self.reach_tree = reach_tree
        self.usafe_set_constraint_list = usafe_set_constraint_list

        self.init_star = reach_tree.nodes[0].state

        self.num_dims = num_dims
        # self.usafe_star = usafe_star

    def compute_usafe_set_pred_in_star_basis(self, error_star):
        usafe_basis_predicates = []

        for std_pred in self.usafe_set_constraint_list:
            # Translating usafe set star into the simulation/star's basis
            new_lc_vector = np.dot(error_star.basis_matrix, std_pred.vector)
            new_lc_value = std_pred.value - np.dot(error_star.center, std_pred.vector)
            usafe_basis_predicates.append(LinearConstraint(new_lc_vector, new_lc_value))

        for constraint in error_star.constraint_list:
            usafe_basis_predicates.append(constraint)

        return usafe_basis_predicates

    @staticmethod
    def convert_star_pred_in_standard_pred(error_star):
        usafe_std_predicates = []
        basis_matrix_inverse = inv(error_star.basis_matrix.T)
        for pred in error_star.constraint_list:
            new_lc_vector = np.dot(pred.vector, basis_matrix_inverse)
            new_lc_value = pred.value + np.dot(pred.vector, np.dot(basis_matrix_inverse, error_star.center))
            usafe_std_predicates.append(LinearConstraint(new_lc_vector, new_lc_value))
        # for pred in self.usafe_set_constraint_list:
        #    usafe_std_predicates.append(pred.clone())
        return usafe_std_predicates

    @staticmethod
    def convert_usafe_basis_pred_in_basis_center(usafe_basis_pred, basis_center):
        offset = np.dot(usafe_basis_pred.vector, basis_center)
        new_val = usafe_basis_pred.value - offset
        new_lc = LinearConstraint(usafe_basis_pred.vector, new_val)

        return new_lc

    def compute_longest_ce_in_a_path(self, path):
        # longest_ce_lpi = None
        direction = np.ones(self.num_dims)
        prev_time_step = path[0].state.total_steps -1
        continuous_patches = []
        current_patch = []
        for node in path:
            if node.state.total_steps == prev_time_step+1:
                current_patch.append(node)
            else:
                continuous_patches.append(current_patch)
                current_patch = [node]
            prev_time_step = node.state.total_steps
            # print('Loc {} time step {}\n'.format(node.state.mode.name, node.state.total_steps))

        if len(current_patch) is not 0:
            continuous_patches.append(current_patch)

        # ce_length = 0
        # ce = None
        ce_object = None
        for patch in continuous_patches:
            # current_ce_length = 0
            # ce = None
            for idx_i in xrange(len(patch)):
                usafe_lpi = LpInstance(self.num_dims, self.num_dims)
                usafe_lpi.update_basis_matrix(self.init_star.basis_matrix)
                prev_node_state = patch[idx_i].state
                basis_centers = []
                current_ce_length = 0

                while True:
                    if isinstance(prev_node_state.parent, InitParent) or isinstance(prev_node_state.parent.star.parent,
                                                                                InitParent):
                        break
                    elif isinstance(prev_node_state.parent.star.parent, DiscretePostParent):
                        basis_centers.append(prev_node_state.parent.star.parent.prestar_basis_center)
                        prev_node_state = prev_node_state.parent.star.parent.prestar

                # TOCHECK Reverse the basis_centers list here?

                prev_node_state = patch[idx_i].state

                for idx_j in range(idx_i, len(patch), 1):
                    node = patch[idx_j]
                    if node.state.mode.name != prev_node_state.mode.name:
                        basis_centers.append(node.state.parent.star.parent.prestar_basis_center)
                        prev_node_state = node.state
                    usafe_basis_preds = self.compute_usafe_set_pred_in_star_basis(node.state)
                    for pred in usafe_basis_preds:
                        for basis_center in basis_centers[::-1]:
                            pred = self.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                        usafe_lpi.add_basis_constraint(pred.vector, pred.value)

                    for pred in self.init_star.constraint_list:
                        usafe_lpi.add_basis_constraint(pred.vector, pred.value)

                    # TOCHECK is this a redundant step?
                    # if isinstance(node.state.parent.star.parent, DiscretePostParent):
                    #    for pred in node.state.parent.star.constraint_list:
                    #        for basis_center in basis_centers[::-1]:
                    #            pred = self.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                    #        usafe_lpi.add_basis_constraint(pred.vector, pred.value)

                    result = np.zeros(self.num_dims)
                    is_feasible = usafe_lpi.minimize(direction, result, error_if_infeasible=False)

                    if is_feasible:
                        feasible_point = np.dot(self.init_star.basis_matrix, result)
                        current_ce_length = current_ce_length + 1
                        current_ce = feasible_point
                        # print "Counterexample is '{}' of length '{}' for node '{}' at time '{}'".format(current_ce,
                        #                            current_ce_length, node.state.mode.name, node.state.total_steps)
                    else:
                        # print "Node '{}' at time '{}' is not feasible with existing states".format(node.state.mode.name,
                        #                                                                        node.state.total_steps)
                        break

                    if ce_object is None or current_ce_length >= ce_object.ce_length:
                        ce_object = CeObject(current_ce, current_ce_length, 0, patch, idx_i, idx_j)
                        # longest_ce_lpi = usafe_lpi
                        # print "This counterexample starts from index '{}' in location '{}'".format(
                        # patch[idx_i].state.total_steps, patch[idx_i].state.mode.name)
                        # print "This counterexample ends at index '{}' in location '{}'".format(
                        # node.state.total_steps, node.state.mode.name)

        return ce_object

    def compute_longest_ce(self, lpi_required=False):
        start_node = self.reach_tree.nodes[0]
        # Timers.tic('New implementation Longest counter-example-1')
        # basis_centers = []
        # constraints_list = []
        # self.check_path_feasibility(start_node, direction, basis_centers, constraints_list)
        # Timers.toc('New implementation Longest counter-example-1')
        paths = []
        current_path = []
        Timers.tic('New implementation Longest counter-example-2 generation time')
        self.explore_tree_for_error_stars(start_node, current_path, paths)
        # print "Number of paths is '{}'".format(len(paths))
        longest_ce = None
        longest_ce_length = 0
        # for node in paths[1]:
        #    print ("Loc: '{}' and time '{}'".format(node.state.mode.name, node.state.total_steps))
        for path in paths:
            ce_object = self.compute_longest_ce_in_a_path(path)
            print(" Paths {} {}\n".format(len(path), ce_object.ce_length))
            # print "Counterexample for this path is '{}' of length '{}'".format(ce, ce_length)
            if ce_object.ce_length >= longest_ce_length:
                longest_ce = ce_object.counterexample
                longest_ce_length = ce_object.ce_length
        # print(" length: {} idx_i: {} idx_j: {}".format(ce_object.ce_length, ce_object.start_index, ce_object.end_index))

        Timers.toc('New implementation Longest counter-example-2 generation time')
        if lpi_required is True:
            return ce_object
        else:
            print ("The longest counter-example is : '{}' with length '{}'".format(longest_ce, longest_ce_length))
            return longest_ce

    '''The method computes those paths containing only error stars in the reach tree'''
    '''We start with an initial node. Because this is a recursive routine, '''
    '''we need to keep track of the current path and an array of all possible paths.'''
    ''''add_path' variable ensures that there is no redundancy. It means that a path'''
    '''is added only when it contains one or more error star(s) than its parent path.'''
    def explore_tree_for_error_stars(self, node, current_path, paths, add_path=False):
        # print("node with loc {} time {}".format(node.state.mode.name, node.state.total_steps))
        if node.error is True:
            current_path.append(node)
        #    print("node added with loc {} time {}".format(node.state.mode.name, node.state.total_steps))
            if add_path is False:
                add_path = True

        if node.cont_transition is None and len(node.disc_transitions) is 0:
            if add_path is True:
                paths.append(current_path)
            return
        else:
            if node.cont_transition is not None:
                path = []
                for current_node in current_path:
                    path.append(current_node)
                current_add_path = add_path
                self.explore_tree_for_error_stars(node.cont_transition.succ_node, path, paths, current_add_path)
            if len(node.disc_transitions) > 0 and node.disc_transitions[0].succ_node.cont_transition is not None:
                path = []
                for current_node in current_path:
                    path.append(current_node)
                current_add_path = False
                self.explore_tree_for_error_stars(node.disc_transitions[0].succ_node.cont_transition.succ_node, path,
                                                  paths, current_add_path)
            elif len(node.disc_transitions) > 0 and node.disc_transitions[0].succ_node.cont_transition is None:
                path = []
                for current_node in current_path:
                    path.append(current_node)
                current_add_path = add_path
                self.explore_tree_for_error_stars(node.disc_transitions[0].succ_node, path, paths, current_add_path)

    @staticmethod
    def convert_preds_into_z3_str(preds):
        z3_constraints = ''
        for pred in preds:
            p_into_list = pred.vector.tolist()
            if z3_constraints is not '':
                z3_constraints = z3_constraints + ', '
            z3_pred_str = ''
            for index in xrange(len(p_into_list)):
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
            for dim in xrange(self.num_dims):
                x_i = 'x_' + str(dim+1)
                z3_preds_file.write('{} = Real(\'{}\')\n'.format(x_i, x_i))
            basis_centers = []

            z3_constraints = self.convert_preds_into_z3_str(init_node.state.constraint_list)
            z3_preds_file.write('init = And({})\n'.format(z3_constraints))
            z3_preds_file.write('s = Solver()\ns.add(init)\n')
            z3_preds_file.write('set_option(rational_to_decimal=True)\n')
            for str_index in xrange(len(regex_string)):
                character = regex_string[str_index]
                node = path[str_index]
                if prev_node_state.mode.name != node.state.mode.name:
                    if isinstance(node.state.parent.star.parent, DiscretePostParent):
                        basis_centers.append(node.state.parent.star.parent.prestar_basis_center)
                        prev_node_state = node.state
                usafe_basis_preds = self.compute_usafe_set_pred_in_star_basis(node.state)
                for pred_index in xrange(len(usafe_basis_preds)):
                    pred = usafe_basis_preds[pred_index]
                    for basis_center in basis_centers[::-1]:
                        pred = self.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                    usafe_basis_preds[pred_index] = pred

                z3_constraints = self.convert_preds_into_z3_str(usafe_basis_preds)
                z3_preds_file.write('b_{} = And({})\n'.format(str_index+1, z3_constraints))
                if character is '1':
                    z3_preds_file.write('s.push()\n')
                    z3_preds_file.write('s.add(b_{})\n'.format(str_index+1))
                elif character is '0':
                    z3_preds_file.write('s.push()\n')
                    z3_preds_file.write('s.add(Not(b_{}))\n'.format(str_index+1))
            z3_preds_file.write('if s.check() == sat:\n')
            z3_preds_file.write('\tm = s.model()\n')
            z3_preds_file.write('\tprint m\n')
            z3_preds_file.write('else:\n')
            z3_preds_file.write('\tprint \'No solution\'')
        # pid = subprocess.Popen([sys.executable, "z3_predicates.py"])  # call subprocess
        env = dict(environ)
        args = ['python', 'z3_predicates.py']
        p = Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        # p = subprocess.Popen('python z3_predicates.py', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout.readlines():
            if line != "No solution\n":
                print ("Counter-example conforming the regular expression ({}) is: {}".format(regex_string, line))
                return line
            else:
                print ("Counter-example conforming the regular expression ({}) is: {}".format(regex_string, "No solution"))
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

    def compute_counter_examples_using_z3(self, str_length):
        # k_strs.append("00111111111111111")
        # k_strs = []
        # k_strs.append("01111100111111111011111111")
        # k_strs.append("11111001111111111111111110")
        # k_strs.append("11111011111111110111111100")

        counter_examples = []
        start_node = self.reach_tree.nodes[0]
        current_path = []
        paths = []
        self.explore_tree_for_error_stars(start_node, current_path, paths)
        path = paths[0]
        Timers.tic("Time taken for computing all string of given length")
        # print (len(path))
        # k_strs_temp = ["".join(seq) for seq in itertools.product("01", repeat=20)]
        Timers.toc("Time taken for computing all string of given length")
        # k_strs = []
        # for i in range(2):
        #    k_strs.append(k_strs_temp[i])

        k_strs = ['00111001111111111111111110'] # Damped Oscillator milp
        # k_strs = ['0011111111101111110']  # Filtered Osc 32
        # k_strs = ['00000001111111111111111110']  # Damped Oscillator
        # k_strs = ['011111111111111']  # Vehicle Platoon 5
        # k_strs = ['11111111111111111111111111111111111111111111111111111111111111111']  # Vehicle Platoon 10
        Timers.tic("Time taken by z3 to find counterexamples for all strings of given length")
        with open("z3_counterexamples", 'w') as z3_ces:
            for str in k_strs:
                current_ce = self.transform_into_z3_problem(str, start_node, path)
                counter_example = np.zeros(self.num_dims)
                if current_ce is not None:
                    z3_ces.write('{}'.format(current_ce))
                    current_ce_tokens = re.split(r'(=|,|\[|\])', current_ce)
                    # print current_ce_tokens
                    for idx in xrange(len(current_ce_tokens)):
                        token = current_ce_tokens[idx]
                        if len(token) > 1:
                            if token[0] is 'x':
                                index = int(token[2:-1])
                                val = current_ce_tokens[idx+2]
                                if val[len(val)-1] is '?':
                                    val = val[:-1]
                                counter_example[index-1] = float(val)
                                idx = idx+2
                            elif token[1] is 'x':
                                index = int(token[3:-1])
                                val = current_ce_tokens[idx+2]
                                if val[len(val)-1] is '?':
                                    val = val[:-1]
                                counter_example[index-1] = float(val)
                                idx = idx + 2
                    counter_examples.append(counter_example)
        z3_ces.close()
        Timers.toc("Time taken by z3 to find counterexamples for all strings of given length")
        return counter_examples

    def dump_path_constraints_for_milp(self):
        start_node = self.reach_tree.nodes[0]
        prev_node_state = start_node.state
        current_path = []
        paths = []
        Timers.tic("Time taken by MILP")
        self.explore_tree_for_error_stars(start_node, current_path, paths)
        for path in paths:

            print ("No of nodes in the path is: '{}'".format(len(path)))
            with open("../milp_preds_files/opt_preds", 'w') as lin_preds_file:

                # lin_preds_file.write('Every set of constraints is represented as Ax <= b, where A is a n*m matrix, x is m*1 vector, and b is a n*1 vector,\n')
                # lin_preds_file.write('where \'n\' is the number of constraints. This set of constraints determine the intersection of the actual symbolic\n')
                # lin_preds_file.write('state and the unsafe region.\n\n')
                # lin_preds_file.write('For a constraint-set, we provide the number of constraints followed by the actual constraints in subsequent rows.\n')
                # lin_preds_file.write('Each constraint-row is designated as the number of non-zero co-efficients, (index, value) pair of each non-zero\n')
                # lin_preds_file.write('co-efficient, and last real-valued number is the corresponding value in vector b.\n\n')
                # lin_preds_file.write('The indices start from 1. For instance, a row \'1 2 1.0 7.0\' determines a constraint 0*x1 + 1*x2 <= 7.\n')
                # lin_preds_file.write('Similarly, \'2 1 6.33 3 0.380 -0.621\' represents the constraint 6.33*x1 + 0*x2 + 0.380*x3 <= -0.621.\n\n\n')
                lin_preds_file.write('{}\n'.format(self.num_dims))
                lin_preds_file.write('{}\n'.format(len(path)))
                basis_centers = []
                for node in path:

                    if prev_node_state.mode.name != node.state.mode.name:
                        if isinstance(node.state.parent.star.parent, DiscretePostParent):
                            basis_centers.append(node.state.parent.star.parent.prestar_basis_center)
                            prev_node_state = node.state
                    usafe_basis_preds = self.compute_usafe_set_pred_in_star_basis(node.state)
                    for index in xrange(len(usafe_basis_preds)):
                        pred = usafe_basis_preds[index]
                        for basis_center in basis_centers[::-1]:
                            pred = self.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                        usafe_basis_preds[index] = pred

                    no_of_constraints = len(usafe_basis_preds)
                    lin_preds_file.write('{}\n'.format(no_of_constraints))
                    for pred in usafe_basis_preds:
                        pred_list = pred.vector.tolist()
                        pred_str = ''
                        no_of_vars_in_pred = 0
                        for index in xrange(len(pred_list)):
                            if pred_list[index] != 0:
                                no_of_vars_in_pred = no_of_vars_in_pred + 1
                                pred_str = pred_str + str(index+1) + ' ' + str(pred_list[index]) + ' '
                        pred_str = pred_str + str(pred.value)
                        lin_preds_file.write('{} {}\n'.format(no_of_vars_in_pred, pred_str))

            env = dict(environ)
            env['LD_LIBRARY_PATH'] = '/home/manish/gurobi810/linux64/lib'
            args = ['/home/manish/gurobi810/linux64/examples/c++/reach_maxz/reach', '../milp_preds_files/opt_preds']
            p = Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            for line in p.stdout.readlines():
                # if line.find("Bounds:") != -1:
                print(line)
            lin_preds_file.close()
        Timers.toc("Time taken by MILP")

    def dump_path_constraints_in_a_file(self):
        start_node = self.reach_tree.nodes[0]
        prev_node_state = start_node.state
        current_path = []
        paths = []
        self.explore_tree(start_node, current_path, paths)
        print ("No of nodes in the path is: '{}'".format(len(paths[0])))
        with open("linear_predicates", 'w') as lin_preds_file:

            lin_preds_file.write('Every set of constraints is represented as Ax <= b, where A is a n*m matrix, x is m*1 vector, and b is a n*1 vector,\n')
            lin_preds_file.write('where \'n\' is the number of constraints. This set of constraints determine the intersection of the actual symbolic\n')
            lin_preds_file.write('state and the unsafe region. This intersection can be empty.\n\n')
            lin_preds_file.write('For a constraint-set, we provide the number of constraints followed by the actual constraints in the subsequent rows.\n')
            lin_preds_file.write('Each constraint-row is designated as the number of non-zero co-efficients, (index, value) pair of each non-zero\n')
            lin_preds_file.write('co-efficient, and the last real-valued number is the corresponding value from the vector b.\n\n')
            lin_preds_file.write('The indices start from 1. For instance, the entry \'1 2 1.0 7.0\' determines a constraint 0*x1 + 1*x2 <= 7.\n')
            lin_preds_file.write('The entry \'2 1 6.33 3 0.380 -0.621\' represents the constraint 6.33*x1 + 0*x2 + 0.380*x3 <= -0.621.\n\n\n')
            basis_centers = []
            for node in paths[0]:

                if prev_node_state.mode.name != node.state.mode.name:
                    if isinstance(node.state.parent.star.parent, DiscretePostParent):
                        basis_centers.append(node.state.parent.star.parent.prestar_basis_center)
                        prev_node_state = node.state
                usafe_basis_preds = self.compute_usafe_set_pred_in_star_basis(node.state)
                for index in xrange(len(usafe_basis_preds)):
                    pred = usafe_basis_preds[index]
                    for basis_center in basis_centers[::-1]:
                        pred = self.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                    usafe_basis_preds[index] = pred

                no_of_constraints = len(usafe_basis_preds)
                lin_preds_file.write('{}\n'.format(no_of_constraints))
                for pred in usafe_basis_preds:
                    pred_list = pred.vector.tolist()
                    pred_str = ''
                    no_of_vars_in_pred = 0
                    for index in xrange(len(pred_list)):
                        if pred_list[index] != 0:
                            no_of_vars_in_pred = no_of_vars_in_pred + 1
                            pred_str = pred_str + str(index+1) + ' ' + str(pred_list[index]) + ' '
                    pred_str = pred_str + str(pred.value)
                    lin_preds_file.write('{} {}\n'.format(no_of_vars_in_pred, pred_str))

    '''The method explores the reach tree to compute all the possible paths'''
    '''We start with an initial node. Because this is a recursive routine, '''
    '''we need to keep track of the current path and an array of all possible paths.'''
    def explore_tree(self, node, current_path, paths):
        current_path.append(node)

        if node.cont_transition is None and len(node.disc_transitions) is 0:
            paths.append(current_path)
            return
        else:
            if node.cont_transition is not None:
                path = []
                for current_node in current_path:
                    path.append(current_node)
                self.explore_tree(node.cont_transition.succ_node, path, paths)
            if len(node.disc_transitions) > 0 and node.disc_transitions[0].succ_node.cont_transition is not None:
                path = []
                for current_node in current_path:
                    path.append(current_node)
                self.explore_tree(node.disc_transitions[0].succ_node.cont_transition.succ_node, path, paths)

    # def convert_point_in_star_basis(self, point, error_star):
    #    basis_predicates = []
    #    identity_matrix = np.identity(self.num_dims, dtype=float)
    #    for dim in range(identity_matrix.ndim):
    #        lc = LinearConstraint(identity_matrix[dim], point[dim])
    #        basis_predicates.append(LinearConstraint(lc.vector, lc.value))
    #        lc = LinearConstraint(-1 * identity_matrix[dim], -point[dim])
    #        basis_predicates.append(LinearConstraint(lc.vector, lc.value))

    #    basis_predicates_in_star_basis = []
    #    for lc in basis_predicates:
    #        lc_vector = lc.vector

    #        new_lc_vector = np.dot(error_star.basis_matrix, lc_vector)
    #        new_lc_value = lc.value - np.dot(error_star.center, lc_vector)
    #        basis_predicates_in_star_basis.append(LinearConstraint(new_lc_vector, new_lc_value))

    #    return basis_predicates_in_star_basis

    @staticmethod
    def convert_std_state_in_star_state(std_point, error_star):
        lc_value = std_point - error_star.center
        basis_matrix_inverse = inv(error_star.basis_matrix.T)
        star_state = np.dot(basis_matrix_inverse, lc_value)
        return star_state

    def compute_deepest_ce(self, direction):
        start_node = self.reach_tree.nodes[0]
        node_queue = [start_node]
        depth = None
        deepest_node = None
        deepest_point = None
        Timers.tic("Deepest counter-example generation time")
        while len(node_queue) is not 0:

            node = node_queue[0]
            node_queue = node_queue[1:]

            if node.error is True:
                usafe_lpi = LpInstance(self.num_dims, self.num_dims)
                # usafe_lpi.update_basis_matrix(np.identity(self.num_dims))
                usafe_lpi.update_basis_matrix(node.state.basis_matrix)
                error_std_preds = self.convert_star_pred_in_standard_pred(node.state)
                for pred in error_std_preds:
                    usafe_lpi.add_standard_constraint(pred.vector, pred.value)
                for pred in self.usafe_set_constraint_list:
                    usafe_lpi.add_standard_constraint(pred.vector, pred.value)
                # usafe_basis_preds = self.compute_usafe_set_pred_in_star_basis(node.state)
                # for pred in node.state.constraint_list:
                #    usafe_lpi.add_basis_constraint(pred.vector, pred.value)
                # for pred in usafe_basis_preds:
                #    usafe_lpi.add_basis_constraint(pred.vector, pred.value)

                result = np.ones(self.num_dims)
                is_feasible = usafe_lpi.minimize(-1 * direction, result, error_if_infeasible=False)
                if is_feasible:
                    # basis_matrix = node.state.basis_matrix
                    basis_matrix = np.identity(self.num_dims, dtype=float)
                    point = np.dot(basis_matrix, result)
                    current_depth = np.dot(direction, point)
                    # print ("depth for the point {} at time {} in loc {} is : {}".format(point, node.state.total_steps, node.state.mode.name, current_depth))
                    if depth is None or current_depth >= depth:
                        depth = current_depth
                        deepest_node = node
                        deepest_point = point

            if node.cont_transition is not None:
                node_queue.append(node.cont_transition.succ_node)
            if len(node.disc_transitions) > 0 and node.disc_transitions[0].succ_node.cont_transition is not None:
                node_queue.append(node.disc_transitions[0].succ_node.cont_transition.succ_node)

        print ("deepest point is '{}' in location '{}' with depth '{}'".format(deepest_point, deepest_node.state.mode.name, depth))
        basis_centers = []
        # basis_matrices = []
        prev_node_state = deepest_node.state
        while True:
            if isinstance(prev_node_state.parent, InitParent) or isinstance(prev_node_state.parent.star.parent,
                                                                            InitParent):
                break
            elif isinstance(prev_node_state.parent.star.parent, DiscretePostParent):
                basis_centers.append(prev_node_state.parent.star.parent.prestar_basis_center)
                # basis_matrices.append(prev_node_state.parent.star.parent.prestar.basis_matrix)
                prev_node_state = prev_node_state.parent.star.parent.prestar

        print ("Basis centers: {}".format(basis_centers))
        # usafe_lpi = LpInstance(self.num_dims, self.num_dims)
        # usafe_lpi.update_basis_matrix(self.init_star.basis_matrix)
        # error_star_state = self.convert_std_state_in_star_state(deepest_point, deepest_node.state)
        error_star_state = deepest_node.state.point_to_star_basis(deepest_point)
        # for index in range(len(basis_centers)-1, -1, -1):
        #    print "Index is {}".format(index)
        #    basis_center = basis_centers[index]
        #    basis_matrix = basis_matrices[index]
        #    basis_center_coeffs = np.dot(inv(basis_matrix.T),basis_center)
        #    error_star_state = error_star_state - basis_center_coeffs
        for basis_center in basis_centers[::-1]:
            error_star_state = error_star_state - basis_center
        deepest_ce = np.dot(self.init_star.basis_matrix.T, error_star_state)
        print ("The deepest ce is '{}' with depth '{}'".format(deepest_ce, depth))
        Timers.toc("Deepest counter-example generation time")
        return deepest_ce

        # usafe_basis_preds = self.compute_usafe_set_pred_in_star_basis(deepest_node.state)
        # all_preds = []
        # for pred in usafe_basis_preds:
        #    for basis_center in basis_centers[::-1]:
        #        pred = self.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
        #    all_preds.append(pred.clone())
        # for pred in deepest_node.state.parent.star.constraint_list:
        #    for basis_center in basis_centers[::-1]:
        #        pred = self.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
        #    all_preds.append(pred.clone())

        # for pred in self.init_star.constraint_list:
        #    all_preds.append(pred.clone())

        # for pred in all_preds:
        #    usafe_lpi.add_basis_constraint(pred.vector, pred.value)
        # result = np.zeros(self.num_dims)
        # is_feasible = usafe_lpi.minimize(direction, result, error_if_infeasible=False)
        # if is_feasible:
        #    feasible_point = np.dot(self.init_star.basis_matrix, result)
        #    deepest_ce = feasible_point
        #    print "The deepest ce is '{}' with depth '{}'".format(deepest_ce, depth)
        # return deepest_ce

    '''Computes robust point in each error star'''
    def compute_robust_ce_old(self):
        start_node = self.reach_tree.nodes[0]
        node_queue = [start_node]
        robust_points = []
        # robust_point = np.zeros(self.num_dims)
        while len(node_queue) is not 0:

            node = node_queue[0]
            node_queue = node_queue[1:]

            if node.error is True:
                usafe_lpi = LpInstance(self.num_dims, self.num_dims)
                usafe_lpi.update_basis_matrix(node.state.basis_matrix)
                for pred in self.usafe_set_constraint_list:
                    usafe_lpi.add_standard_constraint(pred.vector, pred.value)
                for pred in node.state.constraint_list:
                    usafe_lpi.add_basis_constraint(pred.vector, pred.value)

                directions = np.identity(self.num_dims, dtype=float)
                robust_point = np.zeros(self.num_dims)
                for index in xrange(self.num_dims):
                    direction = directions[index]
                    result = np.zeros(self.num_dims)
                    is_feasible = usafe_lpi.minimize(-1 * direction, result, error_if_infeasible=False)
                    if is_feasible:
                        usafe_set_basis_matrix = np.identity(self.num_dims, dtype=float)
                        point1 = np.dot(usafe_set_basis_matrix, result)
                    result = np.zeros(self.num_dims)
                    is_feasible = usafe_lpi.minimize(direction, result, error_if_infeasible=False)
                    if is_feasible:
                        usafe_set_basis_matrix = np.identity(self.num_dims, dtype=float)
                        point2 = np.dot(usafe_set_basis_matrix, result)
                    # print ("Points for the direction {} are {} and {}".format(direction, point1, point2))
                    current_point = (point1 + point2) / 2
                    robust_point[index] = np.dot(current_point, direction)
                print ("Robust point is '{}' in location '{}'".format(robust_point, node.state.mode.name))
                robust_points.append(robust_point)

            if node.cont_transition is not None:
                node_queue.append(node.cont_transition.succ_node)
            if len(node.disc_transitions) > 0 and node.disc_transitions[0].succ_node.cont_transition is not None:
                node_queue.append(node.disc_transitions[0].succ_node.cont_transition.succ_node)

        return robust_points

    '''Computes robust point in each error star, map each of these points to a point in initial star
        and take the average of them.'''
    def compute_robust_ce_new(self):
        Timers.tic('Robust counter-example generation time')
        ce_object = self.compute_longest_ce(True)
        robust_points_in_initial_star = []
        patch = ce_object.patch[1:len(ce_object.patch)-1]
        for node in patch:
            usafe_lpi = LpInstance(self.num_dims, self.num_dims)
            usafe_lpi.update_basis_matrix(node.state.basis_matrix)
            error_std_preds = self.convert_star_pred_in_standard_pred(node.state)
            for pred in error_std_preds:
                usafe_lpi.add_standard_constraint(pred.vector, pred.value)
            for pred in self.usafe_set_constraint_list:
                usafe_lpi.add_standard_constraint(pred.vector, pred.value)

            directions = np.identity(self.num_dims, dtype=float)
            avg_points = []
            for index in xrange(self.num_dims):
                direction = directions[index]
                result = np.ones(self.num_dims)
                is_feasible = usafe_lpi.minimize(direction, result, error_if_infeasible=False)
                if is_feasible:
                    basis_matrix = np.identity(self.num_dims, dtype=float)
                    point1 = np.dot(basis_matrix, result)
                result = np.ones(self.num_dims)
                is_feasible = usafe_lpi.minimize(-1 * direction, result, error_if_infeasible=False)
                if is_feasible:
                    basis_matrix = np.identity(self.num_dims, dtype=float)
                    point2 = np.dot(basis_matrix, result)
                avg_point = (point1 + point2)/2
                avg_points.append(avg_point)
            robust_point_in_star = np.zeros(self.num_dims, dtype=float)
            for point in avg_points:
                robust_point_in_star += point
            robust_point_in_star = robust_point_in_star/len(avg_points)
            # print "robust point in star {}".format(robust_point_in_star)
            basis_centers = []
            prev_node_state = node.state
            while True:
                if isinstance(prev_node_state.parent, InitParent) or isinstance(prev_node_state.parent.star.parent,
                                                                                InitParent):
                    break
                elif isinstance(prev_node_state.parent.star.parent, DiscretePostParent):
                    basis_centers.append(prev_node_state.parent.star.parent.prestar_basis_center)
                    prev_node_state = prev_node_state.parent.star.parent.prestar
            error_star_state = node.state.point_to_star_basis(robust_point_in_star)
            for basis_center in basis_centers[::-1]:
                error_star_state = error_star_state - basis_center
            robust_points_in_initial_star.append(np.dot(self.init_star.basis_matrix.T, error_star_state))

        robust_point = 0.0
        for point in robust_points_in_initial_star:
            robust_point += point
        robust_point = robust_point/len(robust_points_in_initial_star)
        print ("Robust point: '{}'".format(robust_point))
        Timers.toc('Robust counter-example generation time')
        return robust_point

    def create_lpi(self, ce_object):
        longest_usafe_ce_lpi = LpInstance(self.num_dims, self.num_dims)
        longest_usafe_ce_lpi.update_basis_matrix(self.init_star.basis_matrix)
        prev_node_state = ce_object.patch[ce_object.start_index].state
        basis_centers = []
        while True:
            if isinstance(prev_node_state.parent, InitParent) or isinstance(prev_node_state.parent.star.parent,
                                                                            InitParent):
                break
            elif isinstance(prev_node_state.parent.star.parent, DiscretePostParent):
                basis_centers.append(prev_node_state.parent.star.parent.prestar_basis_center)
                prev_node_state = prev_node_state.parent.star.parent.prestar

        for idx in range(ce_object.start_index, ce_object.end_index+1, 1):
            node = ce_object.patch[idx]
            if node.state.mode.name != prev_node_state.mode.name:
                basis_centers.append(node.state.parent.star.parent.prestar_basis_center)
                prev_node_state = node.state
            usafe_basis_preds = self.compute_usafe_set_pred_in_star_basis(node.state)
            for pred in usafe_basis_preds:
                for basis_center in basis_centers[::-1]:
                    pred = self.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                longest_usafe_ce_lpi.add_basis_constraint(pred.vector, pred.value)

        for pred in self.init_star.constraint_list:
            longest_usafe_ce_lpi.add_basis_constraint(pred.vector, pred.value)

        return longest_usafe_ce_lpi

    '''Find a robust point in the initial set after computing the intersection of each error star'''
    def compute_robust_ce(self):
        ce_object = self.compute_longest_ce(True)
        longest_ce_lpi = self.create_lpi(ce_object)
        directions = np.identity(self.num_dims, dtype=float)
        robust_point = np.zeros(self.num_dims)
        for index in xrange(self.num_dims):
            direction = directions[index]
            result = np.zeros(self.num_dims)
            is_feasible = longest_ce_lpi.minimize(-1 * direction, result, error_if_infeasible=False)
            basis_matrix = self.init_star.basis_matrix
            if is_feasible:
                point1 = np.dot(basis_matrix, result)
            result = np.zeros(self.num_dims)
            is_feasible = longest_ce_lpi.minimize(direction, result, error_if_infeasible=False)
            if is_feasible:
                point2 = np.dot(basis_matrix, result)
            # print ("Points for the direction {} are {} and {}".format(direction, point1, point2))
            current_point = (point1 + point2) / 2
            robust_point[index] = np.dot(current_point, direction)
        print ("Robust point is '{}'".format(robust_point))
        return robust_point

    def check_path_feasibility(self, node, direction, basis_centers, constraints_list):
        current_constraints_list = []
        current_basis_centers = []
        for constraint in constraints_list:
            current_constraints_list.append(constraint)
        for basis_center in basis_centers:
            current_basis_centers.append(basis_center)
        if node.error is False:
            print ("Non-error node at '{}' in location '{}'".format(node.state.total_steps, node.state.mode.name))
            if node.cont_transition is not None:
                print (" -- has a continuous transition at '{}' to location '{}'".format(
                    node.cont_transition.succ_node.state.total_steps,
                    node.cont_transition.succ_node.state.mode.name))
                self.check_path_feasibility(node.cont_transition.succ_node, direction, current_basis_centers,
                                            current_constraints_list)
            if len(node.disc_transitions) > 0 and node.disc_transitions[0].succ_node.cont_transition is not None:
                print (" -- has a discrete transition at '{}' to location '{}'".format(
                    node.disc_transitions[0].succ_node.state.total_steps,
                    node.disc_transitions[0].succ_node.state.mode.name))
                basis_centers.append(node.state.center)
                self.check_path_feasibility(node.disc_transitions[0].succ_node, direction, current_basis_centers,
                                            current_constraints_list)
            if node.cont_transition is None and len(node.disc_transitions) == 0:
                print (" -- has no transition. Returning to previous node")
        else:
            print ("Error node at '{}' in location '{}'".format(node.state.total_steps, node.state.mode.name))
            # print "basis centers '{}'".format(basis_centers)
            usafe_lpi = LpInstance(self.num_dims, self.num_dims)
            usafe_lpi.update_basis_matrix(self.init_star.basis_matrix)
            for constraint in current_constraints_list:
                usafe_lpi.add_basis_constraint(constraint.vector, constraint.value)

            usafe_basis_preds = self.compute_usafe_set_pred_in_star_basis(node.state)
            for pred in usafe_basis_preds:
                for basis_center in current_basis_centers[::-1]:
                    pred = self.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                usafe_lpi.add_basis_constraint(pred.vector, pred.value)
                current_constraints_list.append(pred)

            if isinstance(node.state.parent.star.parent, DiscretePostParent):
                for pred in node.state.parent.star.constraint_list:
                    for basis_center in current_basis_centers[::-1]:
                        pred = self.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                    usafe_lpi.add_basis_constraint(pred.vector, pred.value)
                    current_constraints_list.append(pred)

            result = np.zeros(self.num_dims)
            is_feasible = usafe_lpi.minimize(direction, result, error_if_infeasible=False)

            if is_feasible:
                feasible_point = np.dot(self.init_star.basis_matrix, result)
                print ("feasible point is '{}' at the time step '{}'".format(feasible_point, node.state.total_steps))

            if node.cont_transition is None and len(node.disc_transitions) is 0:
                print ("-- has no transition. Returning to previous node")

            if node.cont_transition is not None:
                print (" -- has a continuous transition at '{}' to location '{}'".format(
                    node.cont_transition.succ_node.state.total_steps,
                    node.cont_transition.succ_node.state.mode.name))
                self.check_path_feasibility(node.cont_transition.succ_node, direction, current_basis_centers,
                                            current_constraints_list)
            if len(node.disc_transitions) > 0 and node.disc_transitions[0].succ_node.cont_transition is not None:
                print (" -- '{}' in location '{}' has a discrete transition at '{}' to location '{}'".format(
                    node.state.total_steps, node.state.mode.name,
                    node.disc_transitions[0].succ_node.state.total_steps,
                    node.disc_transitions[0].succ_node.state.mode.name))
                current_basis_centers.append(node.disc_transitions[0].succ_node.state.parent.prestar_basis_center)
                self.check_path_feasibility(node.disc_transitions[0].succ_node.cont_transition.succ_node, direction,
                                            current_basis_centers, current_constraints_list)

    def if_member(self, subset, result_set):
        if len(result_set) is 0:
            return False
        for element in result_set:
            output = set(subset) & set(element)
            if len(output) == len(set(subset)):
                return True
        return False

    def top_down_powerset(self, lst):
        result = None
        start_time = time.clock()
        while True:
            if result is None:
                result = [lst]
            else:
                result = next_result
            #        print ("length is {}".format(len(result[0])))
            abc = len(result[0]) - 1
            next_result = []
            if abc is 0:
                break
            for r_element in result:
                for subset in itertools.combinations(r_element, abc):
                    if self.if_member(subset, next_result) is False:
                        next_result.append(subset)
            print(len(next_result))
        #    print ("Result is : {}".format(result))
        print ("Total time {}".format(time.clock() - start_time))

    def list_powerset(self, lst):
        start_time = time.clock()
        # the power set of the empty set has one element, the empty set
        result = [[]]
        for x in lst:
            # for every additional element in our set the power set consists of the subsets that don't
            # contain this element (just take the previous power set) plus the subsets that do contain
            # the element (use list comprehension to add [x] onto everything in the previous power set)
            # print result
            result.extend([subset + [x] for subset in result])
            # print result
        print ("Time taken by power_set: {}".format(time.clock() - start_time))

    def powerset(self, iterable):
        s = list(iterable)
        start_time = time.clock()
        for comb in combinations(s, len(s)/3):
            i = 1
        # for r in range(len(s), 0, -1):
        #    for comb in combinations(s, r):
        #        print comb
        #        i = 1
        print ("Time taken by power_set reverse: {}".format(time.clock() - start_time))
        return
        # return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))