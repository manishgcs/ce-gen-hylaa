
import numpy as np
import time
from hylaa.glpk_interface import LpInstance
from hylaa.hybrid_automaton import LinearConstraint
from hylaa.counter_example import CounterExample
from hylaa.star import init_hr_to_star
from hylaa.starutil import InitParent, ContinuousPostParent, DiscretePostParent
from hylaa.timerutil import Timers

class PostVerificationObject(object):
    def __init__(self, settings, ha, init, usafe_set_constraint_list, error_stars, reachable_stars=None):

        self.error_stars = error_stars

        self.reachable_stars = reachable_stars

        self.usafe_set_constraint_list = usafe_set_constraint_list

        ## Compute the init star
        self.init_star = init_hr_to_star(settings, init[0][1], init[0][0])

        self.num_dims = len(ha.variables)

        self.step = settings.step
        self.num_steps = settings.num_steps
        self.ha = ha

    def get_error_time_steps_in_mode(self, mode):
        error_time_steps = []

        if len(self.error_stars) > 0:
            for error_star in self.error_stars:
                if error_star.mode.name == mode.name:
                    error_time_steps.append(error_star.total_steps)

        return error_time_steps

    def compute_point_as_star_basis(self, point, error_star):
        basis_predicates = []
        identity_matrix = np.identity(self.num_dims, dtype=float)
        for dim in range(identity_matrix.ndim):
            lc = LinearConstraint(identity_matrix[dim], point[dim])
            basis_predicates.append(LinearConstraint(lc.vector, lc.value))
            lc = LinearConstraint(-1 * identity_matrix[dim], -point[dim])
            basis_predicates.append(LinearConstraint(lc.vector, lc.value))

        basis_predicates_in_star_basis = []
        for lc in basis_predicates:
            lc_vector = lc.vector

            new_lc_vector = np.dot(error_star.basis_matrix, lc_vector)
            new_lc_value = lc.value - np.dot(error_star.center, lc_vector)
            basis_predicates_in_star_basis.append(LinearConstraint(new_lc_vector, new_lc_value))

        return basis_predicates_in_star_basis

    def compute_usafe_basis_pred_in_star_basis(self, error_star, compute_intersection=False):
        usafe_basis_predicates = []

        for usafe_set_lc in self.usafe_set_constraint_list:
            ## To correctly compute the dot product
            lc_vector = usafe_set_lc.vector

            ## Translating usafe set star into the simulation/star's basis
            new_lc_vector = np.dot(error_star.basis_matrix, lc_vector)
            new_lc_value = usafe_set_lc.value - np.dot(error_star.center, lc_vector)
            usafe_basis_predicates.append(LinearConstraint(new_lc_vector, new_lc_value))

        if compute_intersection:
            for constraint in error_star.constraint_list:
                usafe_basis_predicates.append(constraint)

        return usafe_basis_predicates

    def compute_usafe_std_pred_in_star_basis(self, error_star):
        usafe_std_predicates = []

        for usafe_set_lc in self.usafe_set_constraint_list:
            ## To correctly compute the dot product
            lc_t = usafe_set_lc.vector

            new_lc_value = usafe_set_lc.value - np.dot(error_star.center, lc_t)
            usafe_std_predicates.append(LinearConstraint(usafe_set_lc.vector, new_lc_value))

        return usafe_std_predicates

    ## Computes the preimage of a point in the unsafe set

    ## Inputs:
    ## 1) settings
    ## 2) init_star
    ## 3) usafe_star
    def compute_counter_examples(self, direction):

        ## Populate error stars info
        error_star_steps = []
        error_star_modes = []

        for error_star in self.error_stars:
            error_star_steps.append(error_star.total_steps)
            error_star_modes.append(error_star.mode)

        # List of feasible solutions/initial points
        initial_points = []
        usafe_basis_predicates_list = []

        ## Iterate over the stars which intersect with the unsafe set
        ## Eventually there is going to be just one star at a particular
        ## time step that user is interested in.
        for index in xrange(len(self.error_stars)):

            #basis_matrix = error_star_basis_matrices[index]
            #basis_center = error_star_centers[index]
            error_star = self.error_stars[index]
            usafe_basis_predicates = self.compute_usafe_basis_pred_in_star_basis(error_star)

            ## List of predicates for each time step where our standard star intersects with the unsafe set
            ## To be used while computing the longest subsequence.
            usafe_basis_predicates_list.append(usafe_basis_predicates)

            ## Create an LP instance
            usafe_lpi = LpInstance(self.num_dims, self.num_dims)

            ## Update the basis matrix
            usafe_lpi.update_basis_matrix(self.init_star.basis_matrix)

            for predicate in usafe_basis_predicates:
                usafe_lpi.add_basis_constraint(predicate.vector, predicate.value)

            ## Add init star basis constraints to the usafe linear constraints list
            for lc in self.init_star.constraint_list:
                usafe_lpi.add_basis_constraint(lc.vector, lc.value)

            result = np.zeros(self.num_dims)

            is_feasible = usafe_lpi.minimize(direction, result, error_if_infeasible=False)

            ## This gives us a point, if any, in the initial set which leads to an unsafe point at a given time step.
            if is_feasible:
                initial_points.append(np.dot(self.init_star.basis_matrix, result))

        counterExamples = []
        unique_initial_points = []
        for index_i in range(len(initial_points)):
            not_unique = False
            for index_u in range(len(unique_initial_points)):
                out = np.zeros(self.num_dims)
                if (np.subtract(unique_initial_points[index_u], initial_points[index_i]) == out).all():
                    not_unique = True
                    break
            if not not_unique:
                counterExample = CounterExample(initial_points[index_i], usafe_basis_predicates_list, error_star_steps,
                                                error_star_modes, self.num_dims)
                counterExamples.append(counterExample)
                unique_initial_points.append(initial_points[index_i])
        return counterExamples


    def compute_deepest_ce(self, depth_direction):

        compute_intersection = True
        Timers.tic('Deepest counter-example')
        points = []
        for error_star in self.error_stars:
            usafe_lpi = LpInstance(self.num_dims, self.num_dims)
            usafe_lpi.update_basis_matrix(error_star.basis_matrix)
            for pred in self.usafe_set_constraint_list:
                usafe_lpi.add_standard_constraint(pred.vector, pred.value)
            for pred in error_star.constraint_list:
                usafe_lpi.add_basis_constraint(pred.vector, pred.value)
            result = np.zeros(self.num_dims)
            is_feasible = usafe_lpi.minimize(-1 * depth_direction, result, error_if_infeasible=False)
            if is_feasible:
                usafe_set_basis_matrix = np.identity(self.num_dims, dtype=float)
                points.append(np.dot(usafe_set_basis_matrix, result))

        if len(points) is 0:
            print "Result (Deepest counter-example): No solution exists in this direction."
            Timers.toc('Deepest counter-example')
            return

        # Find the index of the error_star corresponding to the max_depth
        max_depth = np.dot(depth_direction, points[0])
        max_depth_error_star = self.error_stars[0].clone()
        max_point = points[0]
        for index in xrange(len(points)):
            point = points[index]
            depth = np.dot(depth_direction, point)
            if depth > max_depth:
               max_depth = depth
               max_depth_error_star = self.error_stars[index].clone()
               max_point = point

        print "The deepest point is: '{}' with max_depth '{}'".format(max_point, max_depth)
        #print "Max error star: '{}'".format(max_depth_error_star)
        #max_depth_error_star = self.error_stars[max_depth_index].clone()

        feasible_point = None
        if max_depth_error_star.mode.name == self.init_star.mode.name:
            basis_matrix = max_depth_error_star.parent.star.basis_matrix
            usafe_lpi = LpInstance(self.num_dims, self.num_dims)
            usafe_lpi.update_basis_matrix(basis_matrix)
            usafe_basis_preds = self.compute_point_as_star_basis(max_point, max_depth_error_star)
            #usafe_basis_preds = self.compute_usafe_basis_pred_in_star_basis(max_depth_error_star, False)
            for pred in usafe_basis_preds:
                usafe_lpi.add_basis_constraint(pred.vector, pred.value)
            for pred in max_depth_error_star.parent.star.constraint_list:
                usafe_lpi.add_basis_constraint(pred.vector, pred.value)

            result = np.zeros(self.num_dims)
            is_feasible = usafe_lpi.minimize(-1 * depth_direction, result, error_if_infeasible=False)
            if is_feasible:
                feasible_point = np.dot(basis_matrix, result)
                print "CounterExample: '{}' with depth '{}' in the given direction".format(feasible_point, max_depth)
            Timers.toc('Deepest counter-example')

        else:
            basis_center = max_depth_error_star.parent.star.parent.prestar_basis_center

            usafe_basis_preds = self.compute_usafe_basis_pred_in_star_basis(max_depth_error_star, compute_intersection)
            #print "Usafe Basis predicates: {}".format(usafe_basis_preds)
            #usafe_basis_preds = self.compute_point_as_star_basis(max_point, max_depth_error_star)
            basis_matrix = max_depth_error_star.parent.star.parent.prestar.parent.star.basis_matrix

            usafe_lpi = LpInstance(self.num_dims, self.num_dims)
            usafe_lpi.update_basis_matrix(basis_matrix)

            all_preds = []
            for pred in usafe_basis_preds:
                all_preds.append(self.convert_usafe_basis_pred_in_basis_center(pred, basis_center))
            #for pred in max_depth_error_star.parent.star.constraint_list:
            #    all_preds.append(self.convert_usafe_basis_pred_in_basis_center(pred, basis_center))

            ## adding constraints from the previous mode initial star (P)
            for pred in max_depth_error_star.parent.star.parent.prestar.parent.star.constraint_list:
                all_preds.append(pred.clone())

            for pred in all_preds:
                usafe_lpi.add_basis_constraint(pred.vector, pred.value)

            result = np.zeros(self.num_dims)
            is_feasible = usafe_lpi.minimize(-1 * depth_direction, result, error_if_infeasible=False)
            if is_feasible:
                feasible_point = np.dot(basis_matrix, result)
                print "CounterExample: '{}' with depth '{}' in the given direction".format(feasible_point, max_depth)
            Timers.toc('Deepest counter-example')
        return feasible_point

    def check_if_feasible(self, usafe_basis_preds, usafe_basis_preds_list_in_first_mode, start_index, end_index, direction):

        ## Create an LP instance
        usafe_lpi = LpInstance(self.num_dims, self.num_dims)
        usafe_lpi.update_basis_matrix(self.init_star.basis_matrix)

        for pred in usafe_basis_preds:
            usafe_lpi.add_basis_constraint(pred.vector, pred.value)

        for idx in xrange(len(usafe_basis_preds_list_in_first_mode)):

            if idx >= start_index and idx <= end_index:
                for pred in usafe_basis_preds_list_in_first_mode[idx]:
                    usafe_lpi.add_basis_constraint(pred.vector, pred.value)

        for pred in self.init_star.constraint_list:
            usafe_lpi.add_basis_constraint(pred.vector, pred.value)

        result = np.zeros(self.num_dims)

        is_feasible = usafe_lpi.minimize(direction, result, error_if_infeasible=False)

        return is_feasible


    ## For a given star in second mode, compute the indices i.e., sequence of feasible error stars in first mode
    ## The output is an array of 2 elements with first element as the start index and second element as the end index of the sequence
    def compute_indices(self, usafe_basis_preds_in_current_mode, usafe_basis_preds_list_in_first_mode, start_index, end_index, direction):

        indices = [-1]
        left_indices = [-1]
        right_indices = [-1]
        if end_index < start_index:
            return indices

        mid = (start_index + end_index) / 2

        is_mid_feasible = self.check_if_feasible(usafe_basis_preds_in_current_mode, usafe_basis_preds_list_in_first_mode, mid,
                                                 mid, direction)

        if start_index != end_index:
            left_indices = self.compute_indices(usafe_basis_preds_in_current_mode, usafe_basis_preds_list_in_first_mode, start_index,
                                            mid - 1, direction)
            right_indices = self.compute_indices(usafe_basis_preds_in_current_mode, usafe_basis_preds_list_in_first_mode, mid + 1,
                                             end_index, direction)

        if not is_mid_feasible:
            indices = left_indices
            indices.append(-1)
            for index in right_indices:
                indices.append(index)
        else:
            if left_indices[len(left_indices) - 1] == (mid - 1) and right_indices[0] == (mid + 1):
                start_idx = left_indices[len(left_indices) - 1]
                end_idx = right_indices[0]
                for idx in reversed(left_indices):
                    if idx == -1:
                        break
                    start_idx = idx
                for idx in right_indices:
                    if idx == -1:
                        break
                    end_idx = idx
                is_feasible = self.check_if_feasible(usafe_basis_preds_in_current_mode, usafe_basis_preds_list_in_first_mode,
                                                     start_idx, end_idx, direction)
                if is_feasible:
                    indices = left_indices
                    indices.append(mid)
                    for index in right_indices:
                        indices.append(index)

            elif (left_indices[len(left_indices) - 1] == (mid - 1)):
                start_idx = left_indices[len(left_indices) - 1]
                for idx in reversed(left_indices):
                    if idx == -1:
                        break
                    start_idx = idx
                end_idx = mid
                is_feasible = self.check_if_feasible(usafe_basis_preds_in_current_mode, usafe_basis_preds_list_in_first_mode,
                                                     start_idx, end_idx, direction)
                if is_feasible:
                    indices = left_indices
                    indices.append(mid)

            elif (right_indices[0] == (mid + 1)):
                start_idx = mid
                end_idx = right_indices[0]
                for idx in right_indices:
                    if idx == -1:
                        break
                    end_idx = idx
                is_feasible = self.check_if_feasible(usafe_basis_preds_in_current_mode, usafe_basis_preds_list_in_first_mode,
                                                     start_idx, end_idx, direction)
                if is_feasible:
                    indices[0] = mid
                    for index in right_indices:
                        indices.append(index)
            else:
                indices[0] = mid

        return indices

    def compute_sequences_intersection(self, first_seq, second_seq):
        intersect = [0, 0]
        if len(first_seq) == 0:
            return second_seq
        elif len(second_seq) == 0:
            return first_seq
        else:
            if first_seq[0] <= second_seq[0]:
                intersect[0] = second_seq[0]
            else:
                intersect[0] = first_seq[0]
            if first_seq[1] <= second_seq[1]:
                intersect[1] = first_seq[1]
            else:
                intersect[1] = second_seq[1]
        return intersect

    def usafe_basis_preds_for_discrete_jumps(self, error_star, compute_intersection):
        discrete_usafe_basis_pred_list = []
        current_parent = error_star.parent
        while True:
            if isinstance(current_parent, ContinuousPostParent):
                current_star = current_parent.star
                usafe_basis_preds = self.compute_usafe_basis_pred_in_star_basis(current_star, compute_intersection)
                discrete_usafe_basis_pred_list.append(usafe_basis_preds)
                current_parent = current_star.parent
            elif isinstance(current_parent, DiscretePostParent):
                current_star = current_parent.prestar
                usafe_basis_preds = self.compute_usafe_basis_pred_in_star_basis(current_star, compute_intersection)
                discrete_usafe_basis_pred_list.append(usafe_basis_preds)
                current_parent = current_star.parent
            elif isinstance(current_parent, InitParent):
                break
        return discrete_usafe_basis_pred_list

    def compute_usafe_basis_pred_in_star_basis_in_star_list(self, error_star_list, compute_intersection):

        usafe_basis_preds_list = []
        for error_star in error_star_list:
            usafe_basis_preds = self.compute_usafe_basis_pred_in_star_basis(error_star, compute_intersection)
            usafe_basis_preds_list.append(usafe_basis_preds)
        return usafe_basis_preds_list

    def compute_usafe_std_pred_in_star_basis_in_star_list(self, error_star_list):
        usafe_std_preds_list = []
        for error_star in error_star_list:
            usafe_std_preds = self.compute_usafe_std_pred_in_star_basis(error_star)
            usafe_std_preds_list.append(usafe_std_preds)
        return usafe_std_preds_list

    def compute_sequence_in_a_mode_wrt_init_mode(self, error_star_list, direction, compute_intersection):

        basis_center = error_star_list[0].parent.star.parent.prestar_basis_center

        usafe_basis_preds_list = self.compute_usafe_basis_pred_in_star_basis_in_star_list(error_star_list,
                                                                                          compute_intersection)
        valid_ids_for_all_indices = []
        feasible_points = []
        basis_matrix = error_star_list[0].parent.star.parent.prestar.parent.star.basis_matrix
        for idx_i in xrange(len(error_star_list)):
            valid_ids_for_current_index = []
            ## Create an LP instance
            usafe_lpi = LpInstance(self.num_dims, self.num_dims)
            usafe_lpi.update_basis_matrix(basis_matrix)

            #if idx_i == 0:
            #    with open("constraints_before", 'w+') as f:
            #        f.write('uasfe_basis_preds\n')
            #        for pred in usafe_basis_preds_list[idx_i]:
            #            f.write('{},{};\n'.format(pred.vector, pred.value))
            #        f.write('parent.star.constraint_list\n')
            #        for pred in error_star_list[0].parent.star.constraint_list:
            #            f.write('{},{};\n'.format(pred.vector, pred.value))
            #        f.write('Initial mode predicates\n')
            #        for pred in error_star_list[0].parent.star.parent.prestar.parent.star.constraint_list:
            #            f.write('{},{};\n'.format(pred.vector, pred.value))

            all_preds = []
            for pred in usafe_basis_preds_list[idx_i]:
                all_preds.append(self.convert_usafe_basis_pred_in_basis_center(pred, basis_center))
            for pred in error_star_list[0].parent.star.constraint_list:
                all_preds.append(self.convert_usafe_basis_pred_in_basis_center(pred, basis_center))

            ## adding constraints from the previous mode initial star (P)
            for pred in error_star_list[0].parent.star.parent.prestar.parent.star.constraint_list:
                all_preds.append(pred.clone())

            for pred in all_preds:
                usafe_lpi.add_basis_constraint(pred.vector, pred.value)

            #if idx_i == 0:
            #    with open("constraints_after", 'w+') as f:
            #        for pred in all_preds:
            #            f.write('{},{};\n'.format(pred.vector, pred.value))

            result = np.zeros(self.num_dims)
            usafe_lpi.minimize(direction, result, error_if_infeasible=False)
            feasible_point = np.dot(basis_matrix, result)
            valid_ids_for_current_index.append(idx_i)
            for idx_j in range(idx_i - 1, -1, -1):
                for pred in usafe_basis_preds_list[idx_j]:
                    new_pred = self.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                    usafe_lpi.add_basis_constraint(new_pred.vector, new_pred.value)

                result = np.zeros(self.num_dims)

                is_feasible = usafe_lpi.minimize(direction, result, error_if_infeasible=False)

                if not is_feasible:
                    break
                valid_ids_for_current_index.append(idx_j)
                feasible_point = np.dot(basis_matrix, result)
            valid_ids_for_all_indices.append(valid_ids_for_current_index)
            feasible_points.append(feasible_point)
        #print "valid ids for current mode: '{}'".format(valid_ids_for_all_indices)
        #print "feasible points: '{}'".format(feasible_points)
        return valid_ids_for_all_indices, feasible_points

    def compute_sequence_in_a_mode(self, error_star_list, direction, compute_intersection):

        usafe_basis_preds_list = self.compute_usafe_basis_pred_in_star_basis_in_star_list(error_star_list, compute_intersection)
        valid_ids_for_all_indices = []
        feasible_points = []

        for idx_i in xrange(len(error_star_list)):
            valid_ids_for_current_index = []
            ## Create an LP instance
            usafe_lpi = LpInstance(self.num_dims, self.num_dims)
            usafe_lpi.update_basis_matrix(error_star_list[0].parent.star.basis_matrix)

            all_preds = []
            for pred in usafe_basis_preds_list[idx_i]:
                all_preds.append(pred.clone())
            for pred in error_star_list[0].parent.star.constraint_list:
                all_preds.append(pred.clone())

            for pred in all_preds:
                usafe_lpi.add_basis_constraint(pred.vector, pred.value)

            result = np.zeros(self.num_dims)
            usafe_lpi.minimize(direction, result, error_if_infeasible=False)
            feasible_point = np.dot(error_star_list[0].parent.star.basis_matrix, result)
            valid_ids_for_current_index.append(idx_i)
            for idx_j in range(idx_i-1, -1, -1):
                for pred in usafe_basis_preds_list[idx_j]:
                    usafe_lpi.add_basis_constraint(pred.vector, pred.value)

                result = np.zeros(self.num_dims)

                is_feasible = usafe_lpi.minimize(direction, result, error_if_infeasible=False)

                if not is_feasible:
                    break
                valid_ids_for_current_index.append(idx_j)
                feasible_point = np.dot(self.init_star.basis_matrix, result)
            valid_ids_for_all_indices.append(valid_ids_for_current_index)
            feasible_points.append(feasible_point)
        return valid_ids_for_all_indices, feasible_points

    def compute_sequences_in_two_modes(self, error_star_list_per_mode, direction, compute_intersection=False):
        error_star_list_second_mode = error_star_list_per_mode[1]
        error_star_list_first_mode = []
        for index in xrange(len(error_star_list_per_mode[0])):
            if (error_star_list_per_mode[0][index].total_steps < error_star_list_second_mode[0].total_steps):
                error_star_list_first_mode.append(error_star_list_per_mode[0][index].clone())
            else:
                break

        usafe_basis_preds_list_first_mode = self.compute_usafe_basis_pred_in_star_basis_in_star_list(error_star_list_first_mode, compute_intersection)
        usafe_basis_preds_list_second_mode = self.compute_usafe_basis_pred_in_star_basis_in_star_list(error_star_list_second_mode, compute_intersection)

        ## compute_sequence_in_a_mode() returns the indices of the feasible stars for each star scanning from left to right
        sequences_in_second_mode, feasible_pts_in_second_mode = self.compute_sequence_in_a_mode_wrt_init_mode(error_star_list_second_mode, direction, compute_intersection)
        sequences_in_first_mode, feasible_pts_in_first_mode = self.compute_sequence_in_a_mode(error_star_list_first_mode, direction, compute_intersection)
        len_of_longest_seq_in_first_mode = 0
        longest_ce_in_first_mode = None
        longest_seq_in_first_mode = [0, 0]
        ce_length_in_first_mode = 0
        for index in xrange(len(sequences_in_first_mode)):
            valid_ids = sequences_in_first_mode[index]
            cur_seq_len = (valid_ids[0]-valid_ids[len(valid_ids)-1])
            if len_of_longest_seq_in_first_mode < cur_seq_len:
                len_of_longest_seq_in_first_mode = cur_seq_len
                longest_seq_in_first_mode[1] = valid_ids[0]
                longest_seq_in_first_mode[0] = valid_ids[len(valid_ids)-1]
                ce_length_in_first_mode = longest_seq_in_first_mode[1] - longest_seq_in_first_mode[0] + 1
                longest_ce_in_first_mode = feasible_pts_in_first_mode[index]

        len_of_longest_seq_in_second_mode = 0
        longest_seq_in_second_mode = [0, 0]
        longest_ce_in_second_mode = None
        ce_length_in_second_mode = 0
        for index in xrange(len(sequences_in_second_mode)):
            valid_ids = sequences_in_second_mode[index]
            cur_seq_len = (valid_ids[0] - valid_ids[len(valid_ids)-1])
            if len_of_longest_seq_in_second_mode < cur_seq_len:
                len_of_longest_seq_in_second_mode = cur_seq_len
                longest_seq_in_second_mode[1] = valid_ids[0]
                longest_seq_in_second_mode[0] = valid_ids[len(valid_ids) - 1]
                ce_length_in_second_mode = longest_seq_in_second_mode[1] - longest_seq_in_second_mode[0] + 1
                longest_ce_in_second_mode = feasible_pts_in_second_mode[index]

        ## Compute indices - Compute longest sequence in first mode for each star in the next mode
        ## The sequence is represented as start and end index. If there is none, both start and end indices are -1
        first_mode_seq_for_each_second_mode_stars = []
        #usafe_basis_preds_list_in_first_mode = self.compute_usafe_basis_pred_in_star_basis_in_star_list(
        #            error_star_list_per_mode[0], compute_intersection)
        for index in xrange(len(error_star_list_second_mode)):

            usafe_basis_preds_for_current_star_in_current_mode = self.compute_usafe_basis_pred_in_star_basis(
                        error_star_list_second_mode[index], compute_intersection)

            basis_center = error_star_list_second_mode[0].parent.star.parent.prestar_basis_center

            new_usafe_basis_preds = []
            for pred in usafe_basis_preds_for_current_star_in_current_mode:
                        new_usafe_basis_preds.append(self.convert_usafe_basis_pred_in_basis_center(pred, basis_center))
            indices = self.compute_indices(new_usafe_basis_preds, usafe_basis_preds_list_first_mode, 0,
                                                   len(usafe_basis_preds_list_first_mode) - 1, direction)
            first_mode_seq_for_each_second_mode_stars.append(self.perform_pruning(indices))

        final_seqs_for_first_mode = []
        final_seqs_for_second_mode = []
        feasible_ces = []
        basis_center = error_star_list_second_mode[0].parent.star.parent.prestar_basis_center
        for valid_ids_for_current_index in sequences_in_second_mode:
            start = 0
            end = len(valid_ids_for_current_index) - 1
            feasible_point = None
            while True:

                ## Create an LP instance
                usafe_lpi = LpInstance(self.num_dims, self.num_dims)
                usafe_lpi.update_basis_matrix(self.init_star.basis_matrix)
                for pred in self.init_star.constraint_list:
                    usafe_lpi.add_basis_constraint(pred.vector, pred.value)

                sequence_1 = first_mode_seq_for_each_second_mode_stars[valid_ids_for_current_index[start]] ## Merging it with same sequence gives the same seq
                for idx in range(start, end, 1 ):
                    sequence_1 = self.compute_sequences_intersection(sequence_1, first_mode_seq_for_each_second_mode_stars[valid_ids_for_current_index[idx]])
                    if sequence_1[1] != -1:
                        for pred in usafe_basis_preds_list_second_mode[valid_ids_for_current_index[idx]]:
                            lc = self.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                            usafe_lpi.add_basis_constraint(lc.vector, lc.value)
                    #sequence_1 = self.compute_sequences_intersection(sequence_1, first_mode_seq_for_each_second_mode_stars[valid_ids_for_current_index[idx]])
                if sequence_1[1] == -1:
                    break
                for idx in xrange(len(sequence_1)):
                    for pred in usafe_basis_preds_list_first_mode[idx]:
                        usafe_lpi.add_basis_constraint(pred.vector, pred.value)

                result = np.zeros(self.num_dims)

                is_feasible = usafe_lpi.minimize(direction, result, error_if_infeasible=False)
                feasible_point = np.dot(self.init_star.basis_matrix, result)
                if is_feasible:
                    break
                else:
                    end = end - 1

            valid_ids = valid_ids_for_current_index[start:end+1] #end+1 for including the last value
            sequence_2 = [-1,-1]
            sequence_2[1] = valid_ids[0]
            sequence_2[0] = valid_ids[len(valid_ids) - 1]
            if sequence_1[1] != -1:
                final_seqs_for_second_mode.append(sequence_2)
                final_seqs_for_first_mode.append(sequence_1)
                feasible_ces.append(feasible_point)

        combined_max_ce_length = 0
        start_idx_first_mode = -1
        start_idx_sec_mode = -1
        end_idx_first_mode = -1
        end_idx_sec_mode = -1
        longest_ce = None
        for index in xrange(len(final_seqs_for_second_mode)):
            current_ce_length = final_seqs_for_first_mode[index][1] - final_seqs_for_first_mode[index][0] + 1
            current_ce_length = current_ce_length + (final_seqs_for_second_mode[index][1] - final_seqs_for_second_mode[index][0] + 1)
            if current_ce_length > combined_max_ce_length:
                start_idx_first_mode = final_seqs_for_first_mode[index][0]
                end_idx_first_mode = final_seqs_for_first_mode[index][1]
                start_idx_sec_mode = final_seqs_for_second_mode[index][0]
                end_idx_sec_mode = final_seqs_for_second_mode[index][1]
                combined_max_ce_length = current_ce_length
                longest_ce = feasible_ces[index]

        if combined_max_ce_length < ce_length_in_first_mode:
            combined_max_ce_length = ce_length_in_first_mode
            longest_ce = longest_ce_in_first_mode
            start_idx_first_mode = longest_ce_in_first_mode[0]
            end_idx_first_mode = longest_ce_in_first_mode[1]
            start_idx_sec_mode = end_idx_sec_mode = -1
        if combined_max_ce_length < ce_length_in_second_mode:
            combined_max_ce_length = ce_length_in_second_mode
            longest_ce = longest_ce_in_second_mode
            start_idx_sec_mode = longest_seq_in_second_mode[0]
            end_idx_sec_mode = longest_seq_in_second_mode[1]
            start_idx_first_mode = end_idx_first_mode = -1

        final_indices = []
        final_indices.append(combined_max_ce_length)
        final_indices.append(start_idx_first_mode)
        final_indices.append(end_idx_first_mode)
        final_indices.append(start_idx_sec_mode)
        final_indices.append(end_idx_sec_mode)

        return final_indices, longest_ce

    # Prune the array if there is any extra -1
    def perform_pruning(self, indices):
        indices.append(-1) #Acts as the delimiter to terminate the loop
        output_indices = []
        output_indices.append(-1)
        output_indices.append(-1)
        valid_sequence = False
        for idx in xrange(len(indices)):
            if indices[idx] != -1 and valid_sequence is False:
                output_indices[0] = indices[idx]
                valid_sequence = True
            elif indices[idx] == -1 and valid_sequence is True:
                output_indices[1] = indices[idx-1]

        return output_indices

    def convert_usafe_basis_pred_in_basis_center(self, usafe_basis_pred, basis_center):
        offset = np.dot(usafe_basis_pred.vector, basis_center)
        new_val = usafe_basis_pred.value - offset
        new_lc = LinearConstraint(usafe_basis_pred.vector, new_val)

        return new_lc

    def compute_longest_ce(self, direction):

        Timers.tic('Longest counter-example')

        compute_intersection = True
        ## Maintain the information of the modes and the respective time steps of their error stars
        error_star_modes = []
        error_star_list_per_mode = []
        error_stars_list_for_prev_mode = []
        error_stars_list_for_prev_mode.append(self.error_stars[0].clone())
        error_star_modes.append(self.error_stars[0].mode.name)
        no_of_modes = 1
        for index in xrange(1, len(self.error_stars), 1):
            error_star = self.error_stars[index]
            if self.error_stars[index].parent.star.total_steps != self.error_stars[index-1].parent.star.total_steps:
                error_star_list_per_mode.append(error_stars_list_for_prev_mode)
                m_found = False
                for m_idx in xrange(len(error_star_modes)):
                    if error_star_modes[m_idx] is error_star.mode.name:
                        m_found = True

                if m_found is False:
                    no_of_modes = no_of_modes + 1

                error_star_modes.append(error_star.mode.name)
                error_stars_list_for_prev_mode = []
                error_stars_list_for_prev_mode.append(error_star.clone())
            else:
                error_stars_list_for_prev_mode.append(error_star.clone())

        error_star_list_per_mode.append(error_stars_list_for_prev_mode)

        if len(error_star_list_per_mode) is 0:
            print "Oops! No counter-example exists"
            Timers.toc('Longest counter-example')
            return

        if no_of_modes is 1:
            ## If only first/initial mode intersects with the unsafe set
            if error_star_modes[0] is self.init_star.mode.name:
                mode_name = error_star_modes[0]
                final_max_length_sequence = []
                final_longest_ce = []
                error_star_list = []
                temp_error_star_list = []
                prev_step = error_star_list_per_mode[0][0].total_steps -1
                for idx in xrange(len(error_star_list_per_mode[0])):
                    error_star = error_star_list_per_mode[0][idx]
                    if error_star.total_steps == prev_step+1:
                        temp_error_star_list.append(error_star)
                        prev_step = error_star.total_steps
                    else:
                        error_star_list.append(temp_error_star_list)
                        temp_error_star_list = []
                        temp_error_star_list.append(error_star)
                        prev_step = error_star.total_steps

                if idx == len(error_star_list_per_mode[0])-1:
                    error_star_list.append(temp_error_star_list)

                error_star_list_per_mode = error_star_list
                for error_star_list_idx in xrange(len(error_star_list_per_mode)):
                    valid_ids_for_all_indices, feasible_ces = self.compute_sequence_in_a_mode(
                        error_star_list_per_mode[error_star_list_idx], direction, compute_intersection)

                    max_length_sequence_for_current_star_list = []
                    longest_ce_for_current_star_list = []
                    for idx in range(len(valid_ids_for_all_indices)-1, -1,-1):
                        if len(valid_ids_for_all_indices[idx]) > len(max_length_sequence_for_current_star_list):
                            max_length_sequence_for_current_star_list = valid_ids_for_all_indices[idx]
                            longest_ce_for_current_star_list = feasible_ces[idx]

                    if len(max_length_sequence_for_current_star_list) > len(final_max_length_sequence):
                        final_max_length_sequence = max_length_sequence_for_current_star_list
                        final_longest_ce = longest_ce_for_current_star_list
                        final_error_star_list_idx = error_star_list_idx

                error_star_list_for_longest_ce = error_star_list_per_mode[final_error_star_list_idx]
                final_seq_end_idx = error_star_list_for_longest_ce[final_max_length_sequence[0]].total_steps
                final_seq_start_idx = error_star_list_for_longest_ce[
                    final_max_length_sequence[len(final_max_length_sequence) - 1]].total_steps
                print "Counter Example: '{}' in mode '{}' stays the longest in unsafe set from '{}' to '{}'".format(
                    final_longest_ce, mode_name, final_seq_start_idx, final_seq_end_idx )

            else:
                ## If only second mode intersects with the unsafe set
                mode_name = error_star_modes[0]
                final_max_length_sequence = []
                final_longest_ce = []
                final_error_star_list_idx = 0
                for error_star_list_idx in xrange(len(error_star_list_per_mode)):
                    valid_ids_for_all_indices, feasible_ces = self.compute_sequence_in_a_mode_wrt_init_mode(
                        error_star_list_per_mode[error_star_list_idx], direction, compute_intersection)
                    max_length_sequence_for_current_star_list = []
                    longest_ce_for_current_star_list = []
                    for idx in xrange(len(valid_ids_for_all_indices)):
                        if len(valid_ids_for_all_indices[idx]) > len(max_length_sequence_for_current_star_list):
                            max_length_sequence_for_current_star_list = valid_ids_for_all_indices[idx]
                            longest_ce_for_current_star_list = feasible_ces[idx]

                    if len(max_length_sequence_for_current_star_list) > len(final_max_length_sequence):
                        final_max_length_sequence = max_length_sequence_for_current_star_list
                        final_longest_ce = longest_ce_for_current_star_list
                        final_error_star_list_idx = error_star_list_idx

                error_star_list_for_longest_ce = error_star_list_per_mode[final_error_star_list_idx]
                final_seq_end_idx = error_star_list_for_longest_ce[final_max_length_sequence[0]].total_steps
                final_seq_start_idx = error_star_list_for_longest_ce[
                    final_max_length_sequence[len(final_max_length_sequence) - 1]].total_steps
                print "Counter Example: '{}' in mode '{}' stays the longest in unsafe set starting from '{}' to '{}' time " \
                      "steps after taking a transition from mode '{}' at time step '{}'".format(
                    final_longest_ce, mode_name, final_seq_start_idx, final_seq_end_idx, self.init_star.mode.name,
                    error_star_list_for_longest_ce[0].parent.star.total_steps)
            Timers.toc('Longest counter-example')
            return final_longest_ce

        error_star_list_first_mode = []
        error_star_list_second_mode = []
        final_indices = None
        final_longest_ce = None
        if len(error_star_list_per_mode) >= 2:
            list_length = len(error_star_list_per_mode)
            error_star_list_first_mode = error_star_list_per_mode[0]
            for index in range(1, list_length,1):
                if final_indices is None or (final_indices[0] < (len(error_star_list_first_mode) + len(error_star_list_per_mode[index]))):
                    error_star_list = []
                    error_star_list.append(error_star_list_first_mode)
                    error_star_list.append(error_star_list_per_mode[index])
                    current_indices, current_longest_ce = self.compute_sequences_in_two_modes(error_star_list, direction, compute_intersection)
                    if final_indices is None or (final_indices[0] < current_indices[0]):
                        final_indices = current_indices
                        final_longest_ce = current_longest_ce
                        error_star_list_second_mode = error_star_list_per_mode[index]

        if final_indices[1] == -1:
            longest_ce_start_step = error_star_list_second_mode[final_indices[3]].total_steps
            longest_ce_end_step = error_star_list_second_mode[final_indices[4]].total_steps
            print "Counter Example '{}' stays from '{}' to '{}' in mode '{}'".format(final_longest_ce, longest_ce_start_step, longest_ce_end_step, error_star_modes[1])
        elif final_indices[3] == -1:
            longest_ce_start_step = error_star_list_first_mode[final_indices[1]].total_steps
            longest_ce_end_step = error_star_list_first_mode[final_indices[2]].total_steps
            print "Counter Example '{}' stays from '{}' to '{}' in mode '{}'".format(final_longest_ce, longest_ce_start_step,
                                                                                     longest_ce_end_step,
                                                                                     error_star_modes[0])
        else:
            print "Counter Example '{}' stays from '{}' to '{}' in mode '{}' and from '{}' to '{}' in mode '{}'".format(final_longest_ce,
                                                                                                                    error_star_list_first_mode[final_indices[1]].total_steps,
                                                                                                                    error_star_list_first_mode[final_indices[2]].total_steps,
                                                                                                                    error_star_modes[0],
                                                                                                                    error_star_list_second_mode[final_indices[3]].total_steps,
                                                                                                                    error_star_list_second_mode[final_indices[4]].total_steps,
                                                                                                                    error_star_modes[1])
        Timers.toc('Longest counter-example')
        return final_longest_ce