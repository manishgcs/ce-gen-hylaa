from farkas_central.farkas_mip import FarkasMIP, check_poly_feasibility
from farkas_central.farkas_smt import FarkasSMT, check_poly_feasibility_smt
from hylaa.starutil import InitParent, DiscretePostParent
import numpy as np
from hylaa.polytope import Polytope
from hylaa.hybrid_automaton import LinearConstraint
from farkas_central.bddGraph import BDDGraphNode, BDDGraph
from hylaa.timerutil import Timers
from memory_profiler import profile
import random


class BDD4CE(object):
    def __init__(self, pv_object, equ_run=True, smt_mip='mip'):
        self.pv_object = pv_object
        self.n_state_vars = pv_object.num_dims
        self.n_paths = 0
        self.p_intersect_u = []
        self.p_intersect_not_u = []
        self.equ_run = equ_run
        self.order_idx = []
        self.smt_mip = smt_mip

    def clear_lists(self):
        self.n_paths = 0
        self.p_intersect_u = []
        self.p_intersect_not_u = []

    def partitiion_error_stars_wrt_usafe_boundary(self):

        if self.n_paths > 0:
            self.clear_lists()

        start_node = self.pv_object.reach_tree.nodes[0]
        paths = []
        current_path = []
        self.pv_object.explore_tree_for_error_stars(start_node, current_path, paths)

        self.n_paths = len(paths)

        for path in paths:
            p_intersect_u_in_path = []
            p_intersect_not_u_in_path = []
            prev_node_state = start_node.state

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
                con_matrix = np.zeros((no_of_constraints, self.n_state_vars), dtype=float)
                rhs = np.zeros(no_of_constraints, dtype=float)
                for idx in range(len(usafe_basis_preds)):
                    pred = usafe_basis_preds[idx]
                    pred_list = pred.vector.tolist()
                    for idy in range(len(pred_list)):
                        if pred_list[idy] != 0:
                            con_matrix[idx][idy] = pred_list[idy]
                    rhs[idx] = pred.value
                # print(con_matrix)
                polytope_u = Polytope(con_matrix, rhs)
                p_intersect_u_in_path.append(polytope_u)

                safe_constraint_list = []
                # print(self.usafe_set_constraint_list)
                for pred in self.pv_object.usafe_set_constraint_list:
                    lc_vector = -1 * pred.vector
                    lc_value = -1 * (pred.value + 0.000001)
                    safe_constraint_list.append(LinearConstraint(lc_vector, lc_value))

                safe_basis_preds = self.pv_object.compute_preds_in_star_basis(safe_constraint_list, node.state)
                for index in range(len(safe_basis_preds)):
                    pred = safe_basis_preds[index]
                    for basis_center in basis_centers[::-1]:
                        # As we are re-using the existing function on safe basis preds, the name might create confusion.
                        pred = self.pv_object.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                    safe_basis_preds[index] = pred

                no_of_constraints = len(safe_basis_preds)
                con_matrix = np.zeros((no_of_constraints, self.n_state_vars), dtype=float)
                rhs = np.zeros(no_of_constraints, dtype=float)
                for idx in range(len(safe_basis_preds)):
                    pred = safe_basis_preds[idx]
                    pred_list = pred.vector.tolist()
                    for idy in range(len(pred_list)):
                        if pred_list[idy] != 0:
                            con_matrix[idx][idy] = pred_list[idy]
                    rhs[idx] = pred.value
                # print(con_matrix)
                polytope_not_u = Polytope(con_matrix, rhs)
                p_intersect_not_u_in_path.append(polytope_not_u)

                # star_basis_preds = []
                # for pred in node.state.constraint_list:
                #     star_basis_preds.append(pred)
                #
                # for index in range(len(star_basis_preds)):
                #     pred = star_basis_preds[index]
                #     for basis_center in basis_centers[::-1]:
                #         # As we are re-using the existing function on safe basis preds, the name might create confusion.
                #         pred = self.pv_object.convert_usafe_basis_pred_in_basis_center(pred, basis_center)
                #     star_basis_preds[index] = pred
                #
                # # print(star_basis_preds)
                # no_of_constraints = len(star_basis_preds)
                # con_matrix = np.zeros((no_of_constraints, self.n_state_vars), dtype=float)
                # rhs = np.zeros(no_of_constraints, dtype=float)
                # for idx in range(len(star_basis_preds)):
                #     pred = star_basis_preds[idx]
                #     pred_list = pred.vector.tolist()
                #     for idy in range(len(pred_list)):
                #         if pred_list[idy] != 0:
                #             con_matrix[idx][idy] = pred_list[idy]
                #     rhs[idx] = pred.value
                # # print(con_matrix)
                # polytope = Polytope(no_of_constraints, con_matrix, rhs)
                # p_in_path.append(polytope)

            self.p_intersect_u.append(p_intersect_u_in_path)
            self.p_intersect_not_u.append(p_intersect_not_u_in_path)

        assert len(self.p_intersect_u) == self.n_paths
        assert len(self.p_intersect_not_u) == self.n_paths

    # test_instance_smt = FarkasSMT(p1, p2, q_set, self.n_state_vars)
    # z_vals, alpha_vals = test_instance_smt.solve_4_both_smt()
    # print(z_vals, alpha_vals)
    # test_instance_smt = FarkasSMT(p2, p1, q_set, self.n_state_vars)
    # z_vals, alpha_vals = test_instance_smt.solve_4_both_smt()
    # print(z_vals, alpha_vals)

    # test_instance_mip.solve_4_one_polytope_mip(poly_idx=2)
    # z_vals, alpha_vals = test_instance_mip.solve_4_both_mip()
    # print(z_vals, alpha_vals)
    # test_instance_mip = FarkasMIP(p2, p1, q_set, self.n_state_vars)
    # z_vals, alpha_vals = test_instance_mip.solve_4_both_mip()
    # print(z_vals, alpha_vals)

    def check_equivalence_among_set(self, poly1, node_set, q1_set, q2_set, level_merge):
        equivalent_node = None

        if len(node_set) == 0 or node_set[0].level != level_merge:
            return equivalent_node

        print("no of nodes in the set " + str(len(node_set)))
        for node2 in node_set:
            print(node2.id)
            z_vals = self.check_equivalence(poly1, node2.polytope, q1_set, q2_set)
            print(z_vals)
            if not z_vals:
                equivalent_node = node2
                break

        return equivalent_node

    def check_equivalence(self, poly1, poly2, q1_set, q2_set):

        # A non-empty return value means two polytopes are not equivalent
        if self.equ_run is False:
            return [1]

        if self.smt_mip == 'mip':
            test_instance_1 = FarkasMIP(poly1, poly2, q1_set, self.n_state_vars)
            test_instance_2 = FarkasMIP(poly2, poly1, q1_set, self.n_state_vars)
            test_instance_3 = FarkasMIP(poly2, poly1, q2_set, self.n_state_vars)
            test_instance_4 = FarkasMIP(poly1, poly2, q2_set, self.n_state_vars)
        else:
            test_instance_1 = FarkasSMT(poly1, poly2, q1_set, self.n_state_vars)
            test_instance_2 = FarkasSMT(poly2, poly1, q1_set, self.n_state_vars)
            test_instance_3 = FarkasSMT(poly2, poly1, q2_set, self.n_state_vars)
            test_instance_4 = FarkasSMT(poly1, poly2, q2_set, self.n_state_vars)

        z_vals, alpha_vals = test_instance_1.solve_4_both()

        print("Here1******************")

        if not z_vals:
            z_vals, alpha_vals = test_instance_2.solve_4_both()

            print("Here2******************")

            if not z_vals:
                z_vals, alpha_vals = test_instance_3.solve_4_both()

                print("Here3******************")

                if not z_vals:
                    z_vals, alpha_vals = test_instance_4.solve_4_both()

                    print("Here4******************")

        return z_vals

    # First implementation where queue_nodes and queue_polytopes are different.
    # Order of node insertion in both queues has to be the same
    # Equivalence is tested between only p and not p at each step.
    # @profile(precision=4)
    def create_bdd_old(self):
        nodes_count = []
        Timers.tic('BDD Construction')

        self.partitiion_error_stars_wrt_usafe_boundary()

        for path_idx in range(self.n_paths):
            p_intersect_u_in_path = self.p_intersect_u[path_idx]
            p_intersect_not_u_in_path = self.p_intersect_not_u[path_idx]
            # p_intersect_u_in_path = p_intersect_u_in_path[::-1]
            # p_intersect_not_u_in_path = p_intersect_not_u_in_path[::-1]

            # r_idx = [2, 0, 1, 3, 4]  # Works for [0, 2, 1, 3, 4] as well
            # p_intersect_u = []
            # p_intersect_not_u = []
            # for idx in r_idx:
            #     p_intersect_u.append(p_intersect_u_in_path[idx])
            #     p_intersect_not_u.append(p_intersect_not_u_in_path[idx])
            #
            # p_intersect_u_in_path = p_intersect_u
            # p_intersect_not_u_in_path = p_intersect_not_u

            n_polytopes = len(p_intersect_u_in_path)
            queue_polytopes = []
            queue_bdd_nodes = []

            bdd_graph = BDDGraph()
            bdd_root = bdd_graph.get_root()

            bdd_node_t0 = BDDGraphNode(node_id='t0', level=n_polytopes, my_regex='t0')
            bdd_node_t1 = BDDGraphNode(node_id='t1', level=n_polytopes, my_regex='t1')

            queue_bdd_nodes.append(bdd_root)
            queue_polytopes.append(Polytope(np.asarray([]), np.asarray([])))  # dummy polytope with no constraints

            current_level = 0

            while current_level < n_polytopes:

                n_nodes_at_current_level = 0  # to incrementally assign id's for labeling the nodes at current level

                p1 = p_intersect_u_in_path[current_level]
                p2 = p_intersect_not_u_in_path[current_level]
                q1_set = p_intersect_u_in_path[(current_level+1):n_polytopes]
                q2_set = p_intersect_not_u_in_path[(current_level+1):n_polytopes]

                while queue_bdd_nodes and queue_bdd_nodes[0].level == current_level:
                    current_node = queue_bdd_nodes.pop(0)
                    current_node_regex = current_node.my_regex  # regex is just for debugging and printing the path
                    current_p = queue_polytopes.pop(0)

                    p_intersect_p1 = current_p.intersect_w_q(p1)
                    p_intersect_p2 = current_p.intersect_w_q(p2)

                    # To check, whether either or both polytopes are infeasible
                    alpha_vals1 = check_poly_feasibility(p_intersect_p1)
                    alpha_vals2 = check_poly_feasibility(p_intersect_p2)

                    # If both are infeasible
                    if not alpha_vals1 and not alpha_vals2:
                        print(" \n **** Both {} and {} are in-feasible **** \n".format((current_node_regex + '1'),
                                                                                       (current_node_regex + '0')))
                        current_node.new_transition(bdd_node_t0, 1)
                        current_node.new_transition(bdd_node_t0, 0)
                        continue
                    elif not alpha_vals1:  # If first is infeasible
                        print(" \n **** {} is in-feasible **** \n".format((current_node_regex + '1')))
                        current_node.new_transition(bdd_node_t0, 1)  # Make transition on label 1 to terminal 0

                        if current_level != n_polytopes - 1:  # Add these nodes/polytopes only if not the last level
                            node_id = str(current_level + 1) + str(n_nodes_at_current_level)
                            n_nodes_at_current_level = n_nodes_at_current_level + 1
                            bdd_node = BDDGraphNode(node_id=node_id, level=current_level + 1,
                                                    my_regex=current_node_regex + '0')
                            current_node.new_transition(bdd_node, 0)
                            queue_polytopes.append(p_intersect_p2)
                            queue_bdd_nodes.append(bdd_node)
                        else:
                            print(" \n **** {} is feasible **** \n".format((current_node_regex + '0')))
                            current_node.new_transition(bdd_node_t1, 0)  # Make transition on label 0 to terminal 1
                        continue
                    elif not alpha_vals2:  # If second is infeasible
                        print(" \n **** {} is in-feasible **** \n".format((current_node_regex + '0')))
                        current_node.new_transition(bdd_node_t0, 0)  # Make transition on label 0 to terminal 0

                        if current_level != n_polytopes - 1:  # Add these nodes/polytopes only if not the last level
                            node_id = str(current_level + 1) + str(n_nodes_at_current_level)
                            n_nodes_at_current_level = n_nodes_at_current_level + 1
                            bdd_node = BDDGraphNode(node_id=node_id, level=current_level + 1,
                                                    my_regex=current_node_regex + '1')
                            current_node.new_transition(bdd_node, 1)
                            queue_polytopes.append(p_intersect_p1)
                            queue_bdd_nodes.append(bdd_node)
                        else:
                            print(" \n **** {} is in-feasible **** \n".format((current_node_regex + '1')))
                            current_node.new_transition(bdd_node_t1, 1)  # Make transition on label 1 to terminal 1
                        continue
                    else:
                        # Reach here only when both polytopes are feasible
                        print(" \n **** Both {} and {} are feasible **** \n".format((current_node_regex + '1'),
                                                                                    (current_node_regex + '0')))
                        if current_level == n_polytopes - 1:  # terminal case
                            current_node.new_transition(bdd_node_t1, 0)  # Make transition on label 0 to terminal 1
                            current_node.new_transition(bdd_node_t1, 1)  # Make transition on label 1 to terminal 1
                            continue

                    z_vals = self.check_equivalence(p_intersect_p1, p_intersect_p2, q1_set, q2_set)

                    if not z_vals:
                        print(" \n **** Both {} and {} are equivalent **** \n".format((current_node_regex + '1'),
                                                                                      (current_node_regex + '0')))

                        node_id = str(current_level + 1) + str(n_nodes_at_current_level)
                        n_nodes_at_current_level = n_nodes_at_current_level + 1
                        bdd_node = BDDGraphNode(node_id=node_id, level=current_level + 1,
                                                my_regex=current_node_regex + '2')
                        current_node.new_transition(bdd_node, 1)
                        queue_bdd_nodes.append(bdd_node)
                        queue_polytopes.append(p_intersect_p1)
                        # current_node.new_transition(bdd_node, 0)
                    else:
                        print(z_vals)
                        node_id = str(current_level+1)+str(n_nodes_at_current_level)
                        n_nodes_at_current_level = n_nodes_at_current_level + 1
                        bdd_node_1 = BDDGraphNode(node_id=node_id, level=current_level+1,
                                                  my_regex=current_node_regex + '1')
                        current_node.new_transition(bdd_node_1, 1)
                        node_id = str(current_level+1)+str(n_nodes_at_current_level)
                        n_nodes_at_current_level = n_nodes_at_current_level + 1
                        bdd_node_0 = BDDGraphNode(node_id=node_id, level=current_level+1,
                                                  my_regex=current_node_regex + '0')
                        current_node.new_transition(bdd_node_0, 0)

                        # Ordering of adding nodes to the list matters until we make polytope as a member of the node
                        queue_bdd_nodes.append(bdd_node_1)
                        queue_bdd_nodes.append(bdd_node_0)
                        queue_polytopes.append(p_intersect_p1)
                        queue_polytopes.append(p_intersect_p2)

                current_level = current_level + 1
                print(" current level is " + str(current_level))
                # print(len(queue_bdd_nodes))
                nodes_count.append(len(queue_bdd_nodes))
                # for idx in range(len(queue_bdd_nodes)):
                #     print(" node regex " + queue_bdd_nodes[idx].my_regex)
        if nodes_count[len(nodes_count) - 1] == 0:  # terminal nodes
            nodes_count[len(nodes_count)-1] = 2
        print(nodes_count)
        Timers.toc('BDD Construction')

    # The polytope is made a member of BDD node
    # Equivalence is tested between only p and not p at each step.
    # @profile(precision=4)
    def create_bdd(self):
        nodes_count = []
        unique_states = []
        Timers.tic('BDD Construction New')

        self.partitiion_error_stars_wrt_usafe_boundary()

        bdd_graphs = []

        for path_idx in range(self.n_paths):
            p_intersect_u_in_path = self.p_intersect_u[path_idx]
            p_intersect_not_u_in_path = self.p_intersect_not_u[path_idx]
            p_intersect_u_in_path = p_intersect_u_in_path[::-1]
            p_intersect_not_u_in_path = p_intersect_not_u_in_path[::-1]

            # r_idx = [2, 0, 1, 3, 4]  # Works for [0, 2, 1, 3, 4] as well
            # p_intersect_u = []
            # p_intersect_not_u = []
            # for idx in r_idx:
            #     p_intersect_u.append(p_intersect_u_in_path[idx])
            #     p_intersect_not_u.append(p_intersect_not_u_in_path[idx])
            #
            # p_intersect_u_in_path = p_intersect_u
            # p_intersect_not_u_in_path = p_intersect_not_u

            n_polytopes = len(p_intersect_u_in_path)
            queue_bdd_nodes = []

            bdd_graph = BDDGraph()
            bdd_root = bdd_graph.get_root()

            bdd_node_t0 = BDDGraphNode(node_id='t0', level=n_polytopes, my_regex='t0')
            bdd_node_t1 = BDDGraphNode(node_id='t1', level=n_polytopes, my_regex='t1')

            queue_bdd_nodes.append(bdd_root)

            current_level = 0

            while current_level < n_polytopes:

                n_nodes_at_current_level = 0  # to incrementally assign id's for labeling the nodes at current level

                p1 = p_intersect_u_in_path[current_level]
                not_p1 = p_intersect_not_u_in_path[current_level]
                q1_set = p_intersect_u_in_path[(current_level+1):n_polytopes]
                q2_set = p_intersect_not_u_in_path[(current_level+1):n_polytopes]

                while queue_bdd_nodes and queue_bdd_nodes[0].level == current_level:
                    current_node = queue_bdd_nodes.pop(0)
                    current_node_regex = current_node.my_regex  # regex is just for debugging and printing the path
                    current_p = current_node.polytope

                    p_intersect_p1 = current_p.intersect_w_q(p1)
                    p_intersect_not_p1 = current_p.intersect_w_q(not_p1)

                    # To check, whether either or both polytopes are infeasible
                    alpha_vals1 = check_poly_feasibility(p_intersect_p1)
                    alpha_vals2 = check_poly_feasibility(p_intersect_not_p1)

                    # If both are infeasible
                    if not alpha_vals1 and not alpha_vals2:
                        print(" \n **** Both {} and {} are in-feasible **** \n".format((current_node_regex + '1'),
                                                                                       (current_node_regex + '0')))
                        current_node.new_transition(bdd_node_t0, 1)
                        current_node.new_transition(bdd_node_t0, 0)
                        continue
                    elif not alpha_vals1:  # If first is infeasible
                        print(" \n **** {} is in-feasible **** \n".format((current_node_regex + '1')))
                        current_node.new_transition(bdd_node_t0, 1)  # Make transition on label 1 to terminal 0

                        if current_level != n_polytopes - 1:  # Add these nodes/polytopes only if not the last level
                            node_id = str(current_level + 1) + str(n_nodes_at_current_level)
                            n_nodes_at_current_level = n_nodes_at_current_level + 1
                            bdd_node = BDDGraphNode(node_id=node_id, level=current_level + 1,
                                                    my_regex=current_node_regex + '0', poly=p_intersect_not_p1)
                            current_node.new_transition(bdd_node, 0)
                            queue_bdd_nodes.append(bdd_node)
                        else:
                            print(" \n **** {} is feasible **** \n".format((current_node_regex + '0')))
                            unique_states.append(alpha_vals2)
                            current_node.new_transition(bdd_node_t1, 0)  # Make transition on label 0 to terminal 1
                        continue
                    elif not alpha_vals2:  # If second is infeasible
                        print(" \n **** {} is in-feasible **** \n".format((current_node_regex + '0')))
                        current_node.new_transition(bdd_node_t0, 0)  # Make transition on label 0 to terminal 0

                        if current_level != n_polytopes - 1:  # Add these nodes/polytopes only if not the last level
                            node_id = str(current_level + 1) + str(n_nodes_at_current_level)
                            n_nodes_at_current_level = n_nodes_at_current_level + 1
                            bdd_node = BDDGraphNode(node_id=node_id, level=current_level + 1,
                                                    my_regex=current_node_regex + '1', poly=p_intersect_p1)
                            current_node.new_transition(bdd_node, 1)
                            queue_bdd_nodes.append(bdd_node)
                        else:
                            print(" \n **** {} is in-feasible **** \n".format((current_node_regex + '1')))
                            unique_states.append(alpha_vals1)
                            current_node.new_transition(bdd_node_t1, 1)  # Make transition on label 1 to terminal 1
                        continue
                    else:
                        # Reach here only when both polytopes are feasible
                        print(" \n **** Both {} and {} are feasible **** \n".format((current_node_regex + '1'),
                                                                                    (current_node_regex + '0')))
                        if current_level == n_polytopes - 1:  # terminal case
                            unique_states.append(alpha_vals1)
                            unique_states.append(alpha_vals2)
                            current_node.new_transition(bdd_node_t1, 0)  # Make transition on label 0 to terminal 1
                            current_node.new_transition(bdd_node_t1, 1)  # Make transition on label 1 to terminal 1
                            continue

                    z_vals = self.check_equivalence(p_intersect_p1, p_intersect_not_p1, q1_set, q2_set)
                    z_vals_second = self.check_equivalence(p_intersect_p1, p_intersect_not_p1, q1_set, q2_set, True)

                    if not z_vals and not z_vals_second:
                        print(" \n **** Both {} and {} are equivalent **** \n".format((current_node_regex + '1'),
                                                                                      (current_node_regex + '0')))

                        node_id = str(current_level + 1) + str(n_nodes_at_current_level)
                        n_nodes_at_current_level = n_nodes_at_current_level + 1
                        bdd_node = BDDGraphNode(node_id=node_id, level=current_level + 1,
                                                my_regex=current_node_regex + '2', poly=p_intersect_p1)
                        current_node.new_transition(bdd_node, 1)
                        queue_bdd_nodes.append(bdd_node)
                        current_node.new_transition(bdd_node, 0)
                    else:
                        print(z_vals)
                        node_id = str(current_level+1)+str(n_nodes_at_current_level)
                        n_nodes_at_current_level = n_nodes_at_current_level + 1
                        bdd_node_1 = BDDGraphNode(node_id=node_id, level=current_level+1,
                                                  my_regex=current_node_regex + '1', poly=p_intersect_p1)
                        current_node.new_transition(bdd_node_1, 1)
                        node_id = str(current_level+1)+str(n_nodes_at_current_level)
                        n_nodes_at_current_level = n_nodes_at_current_level + 1
                        bdd_node_0 = BDDGraphNode(node_id=node_id, level=current_level+1,
                                                  my_regex=current_node_regex + '0', poly=p_intersect_not_p1)
                        current_node.new_transition(bdd_node_0, 0)

                        queue_bdd_nodes.append(bdd_node_1)
                        queue_bdd_nodes.append(bdd_node_0)

                current_level = current_level + 1
                print(" current level is " + str(current_level))
                nodes_count.append(len(queue_bdd_nodes))
                # for idx in range(len(queue_bdd_nodes)):
                #     print(" node regex " + queue_bdd_nodes[idx].my_regex)
            bdd_graphs.append(bdd_graph)
        if nodes_count[len(nodes_count) - 1] == 0:  # terminal nodes
            nodes_count[len(nodes_count)-1] = 2
        print(nodes_count)
        # print(unique_states)
        Timers.toc('BDD Construction New')
        return bdd_graphs

    # Equivalence is checked between p and not p at each step. Additionally, the bdd is tested for equivalence
    # among all nodes at a user provided level.
    # @profile(precision=4)
    def create_bdd_w_level_merge(self, level_merge=-1, order='default'):
        nodes_count = []
        unique_states = []
        Timers.tic('BDD Construction New')

        self.partitiion_error_stars_wrt_usafe_boundary()
        bdd_graphs = []

        for path_idx in range(self.n_paths):

            p_intersect_u_in_path = self.p_intersect_u[path_idx]
            p_intersect_not_u_in_path = self.p_intersect_not_u[path_idx]

            n_polytopes = len(p_intersect_u_in_path)
            if level_merge >= n_polytopes:
                print("\n ******** Please specify a level between 1 and {} for merging *********** \n ".format(str(n_polytopes-1)))
                continue

            order_idx = list(range(n_polytopes))

            if order == 'reverse':
                order_idx = order_idx[::-1]
            elif order == 'random':
                random.Random(20).shuffle(order_idx)
                # r_idx = random.sample(r_idx, len(r_idx))

            self.order_idx = order_idx

            # r_idx = [2, 0, 1, 3, 4]  # Works for [0, 2, 1, 3, 4] as well
            # order_idx = [2, 0, 1, 3, 4]

            if order != 'default':
                print(order_idx)
                p_intersect_u = []
                p_intersect_not_u = []
                for idx in order_idx:
                    p_intersect_u.append(p_intersect_u_in_path[idx])
                    p_intersect_not_u.append(p_intersect_not_u_in_path[idx])

                p_intersect_u_in_path = p_intersect_u
                p_intersect_not_u_in_path = p_intersect_not_u

            queue_bdd_nodes = []

            bdd_graph = BDDGraph()
            bdd_root = bdd_graph.get_root()

            bdd_node_t0 = BDDGraphNode(node_id='t0', level=n_polytopes, my_regex='t0')
            bdd_node_t1 = BDDGraphNode(node_id='t1', level=n_polytopes, my_regex='t1')

            queue_bdd_nodes.append(bdd_root)

            current_level = 0

            while current_level < n_polytopes:

                n_nodes_at_next_level = 0  # to incrementally assign id's for labeling the nodes at current level
                nodes_at_next_level = []

                p1 = p_intersect_u_in_path[current_level]
                not_p1 = p_intersect_not_u_in_path[current_level]
                q1_set = p_intersect_u_in_path[(current_level+1):n_polytopes]
                q2_set = p_intersect_not_u_in_path[(current_level+1):n_polytopes]

                while queue_bdd_nodes and queue_bdd_nodes[0].level == current_level:
                    current_node = queue_bdd_nodes.pop(0)
                    current_node_regex = current_node.my_regex  # regex is just for debugging and printing the path
                    current_p = current_node.polytope

                    p_intersect_p1 = current_p.intersect_w_q(p1)
                    p_intersect_not_p1 = current_p.intersect_w_q(not_p1)

                    # To check, whether either or both polytopes are infeasible

                    if self.smt_mip == 'smt':
                        alpha_vals1 = check_poly_feasibility_smt(p_intersect_p1)
                        alpha_vals2 = check_poly_feasibility_smt(p_intersect_not_p1)
                    else:
                        alpha_vals1 = check_poly_feasibility(p_intersect_p1)
                        alpha_vals2 = check_poly_feasibility(p_intersect_not_p1)

                    # If both are infeasible
                    if not alpha_vals1 and not alpha_vals2:
                        print(" \n **** Both {} and {} are in-feasible **** \n".format((current_node_regex + '1'),
                                                                                       (current_node_regex + '0')))
                        current_node.new_transition(bdd_node_t0, 1)
                        current_node.new_transition(bdd_node_t0, 0)
                        continue
                    elif not alpha_vals1:  # If first is infeasible
                        print(" \n **** {} is in-feasible **** \n".format((current_node_regex + '1')))

                        # Make transition on label 1 to terminal 0 because transition on 1 is infeasible
                        current_node.new_transition(bdd_node_t0, 1)

                        if current_level != n_polytopes - 1:  # Add these nodes/polytopes only if not the last level

                            equ_node = None
                            if level_merge != -1:
                                equ_node = self.check_equivalence_among_set(p_intersect_not_p1, nodes_at_next_level, q1_set, q2_set, level_merge)

                            if equ_node is None:
                                print("equivalent node not found")
                                node_id = str(current_level + 1) + str(n_nodes_at_next_level)
                                n_nodes_at_next_level = n_nodes_at_next_level + 1
                                bdd_node = BDDGraphNode(node_id=node_id, level=current_level + 1,
                                                        my_regex=current_node_regex + '0', poly=p_intersect_not_p1)
                                current_node.new_transition(bdd_node, 0)

                                queue_bdd_nodes.append(bdd_node)

                                if level_merge != -1:
                                    nodes_at_next_level.append(bdd_node)
                            else:
                                print("equivalent node is found")
                                current_node.new_transition(equ_node, 0)
                        else:  # terminal level
                            print(" \n **** {} is feasible **** \n".format((current_node_regex + '0')))
                            unique_states.append(alpha_vals2)
                            current_node.new_transition(bdd_node_t1, 0)  # Make transition on label 0 to terminal 1
                        continue
                    elif not alpha_vals2:  # If second is infeasible
                        print(" \n **** {} is in-feasible **** \n".format((current_node_regex + '0')))
                        current_node.new_transition(bdd_node_t0, 0)  # Make transition on label 0 to terminal 0

                        if current_level != n_polytopes - 1:  # Add these nodes/polytopes only if not the last level

                            equ_node = None
                            if level_merge != -1:
                                equ_node = self.check_equivalence_among_set(p_intersect_p1, nodes_at_next_level,
                                                                            q1_set, q2_set, level_merge)

                            if equ_node is None:
                                print(" equivalent node not found ")
                                node_id = str(current_level + 1) + str(n_nodes_at_next_level)
                                n_nodes_at_next_level = n_nodes_at_next_level + 1
                                bdd_node = BDDGraphNode(node_id=node_id, level=current_level + 1,
                                                        my_regex=current_node_regex + '1', poly=p_intersect_p1)
                                current_node.new_transition(bdd_node, 1)
                                queue_bdd_nodes.append(bdd_node)

                                if level_merge != -1:
                                    nodes_at_next_level.append(bdd_node)
                            else:
                                print("equivalent node is found")
                                current_node.new_transition(equ_node, 1)
                        else:  # terminal level
                            print(" \n **** {} is in-feasible **** \n".format((current_node_regex + '1')))
                            unique_states.append(alpha_vals1)
                            current_node.new_transition(bdd_node_t1, 1)  # Make transition on label 1 to terminal 1
                        continue
                    else:
                        # Reach here only when both polytopes are feasible
                        print(" \n **** Both {} and {} are feasible **** \n".format((current_node_regex + '1'),
                                                                                    (current_node_regex + '0')))
                        if current_level == n_polytopes - 1:  # terminal case
                            unique_states.append(alpha_vals1)
                            unique_states.append(alpha_vals2)
                            current_node.new_transition(bdd_node_t1, 0)  # Make transition on label 0 to terminal 1
                            current_node.new_transition(bdd_node_t1, 1)  # Make transition on label 1 to terminal 1
                            continue

                    z_vals = self.check_equivalence(p_intersect_p1, p_intersect_not_p1, q1_set, q2_set)
                    # z_vals_second = self.check_equivalence(p_intersect_p1, p_intersect_not_p1, q1_set, q2_set, True)

                    if not z_vals:
                        print(" \n **** Both {} and {} are equivalent **** \n".format((current_node_regex + '1'),
                                                                                      (current_node_regex + '0')))

                        equ_node = None
                        if level_merge != -1:
                            equ_node = self.check_equivalence_among_set(p_intersect_p1, nodes_at_next_level, q1_set,
                                                                        q2_set, level_merge)

                        if equ_node is None:
                            print("equivalent node not found")
                            node_id = str(current_level + 1) + str(n_nodes_at_next_level)
                            n_nodes_at_next_level = n_nodes_at_next_level + 1
                            bdd_node = BDDGraphNode(node_id=node_id, level=current_level + 1,
                                                    my_regex=current_node_regex + '2', poly=p_intersect_p1)
                            current_node.new_transition(bdd_node, 1)
                            current_node.new_transition(bdd_node, 0)
                            queue_bdd_nodes.append(bdd_node)

                            if level_merge != -1:
                                nodes_at_next_level.append(bdd_node)
                        else:
                            print("*************** Found an equivalent node in the set ****************")
                            current_node.new_transition(equ_node, 1)
                            current_node.new_transition(equ_node, 0)
                    else:
                        print(z_vals)

                        equ_node = None
                        if level_merge != -1:
                            equ_node = self.check_equivalence_among_set(p_intersect_p1, nodes_at_next_level, q1_set,
                                                                        q2_set, level_merge)

                        if equ_node is None:
                            print("equivalent node not found")
                            node_id = str(current_level+1)+str(n_nodes_at_next_level)
                            n_nodes_at_next_level = n_nodes_at_next_level + 1
                            bdd_node_1 = BDDGraphNode(node_id=node_id, level=current_level+1,
                                                      my_regex=current_node_regex + '1', poly=p_intersect_p1)
                            current_node.new_transition(bdd_node_1, 1)
                            queue_bdd_nodes.append(bdd_node_1)

                            if level_merge != -1:
                                nodes_at_next_level.append(bdd_node_1)
                        else:
                            print("*************** Found an equivalent node in the set ****************")
                            current_node.new_transition(equ_node, 1)

                        equ_node = None
                        if level_merge != -1:
                            equ_node = self.check_equivalence_among_set(p_intersect_not_p1, nodes_at_next_level, q1_set,
                                                                        q2_set, level_merge)
                        if equ_node is None:
                            print("equivalent node not found")
                            node_id = str(current_level+1)+str(n_nodes_at_next_level)
                            n_nodes_at_next_level = n_nodes_at_next_level + 1
                            bdd_node_0 = BDDGraphNode(node_id=node_id, level=current_level+1,
                                                      my_regex=current_node_regex + '0', poly=p_intersect_not_p1)
                            current_node.new_transition(bdd_node_0, 0)
                            queue_bdd_nodes.append(bdd_node_0)

                            if level_merge != -1:
                                nodes_at_next_level.append(bdd_node_0)
                        else:
                            print("*************** Found an equivalent node in the set ****************")
                            current_node.new_transition(equ_node, 0)

                current_level = current_level + 1
                print(" current level is " + str(current_level))
                nodes_count.append(len(queue_bdd_nodes))

            bdd_graphs.append(bdd_graph)
        if nodes_count and nodes_count[len(nodes_count) - 1] == 0:  # terminal nodes
            nodes_count[len(nodes_count)-1] = 2

        if nodes_count:
            print(nodes_count, sum(nodes_count))

            # print(r_idx)
        # print(unique_states)
        Timers.toc('BDD Construction New')
        return bdd_graphs
