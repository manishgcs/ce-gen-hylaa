'''
Main Hylaa Reachability Implementation
Stanley Bak
Aug 2016
'''

from collections import OrderedDict
import numpy as np

from hylaa.plotutil import PlotManager
from hylaa.star import Star
from hylaa.star import init_hr_to_star, init_constraints_to_star
from hylaa.starutil import InitParent, AggregationParent, ContinuousPostParent, DiscretePostParent, make_aggregated_star
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearAutomatonMode, LinearConstraint, HyperRectangle
from hylaa.timerutil import Timers
from hylaa.containers import HylaaSettings, PlotSettings, HylaaResult
from hylaa.glpk_interface import LpInstance
from hylaa.reach_tree import ReachTree, ReachTreeNode, ReachTreeTransition
import hylaa.openblas as openblas


class HylaaEngine(object):
    'main computation object. initialize and call run()'

    def __init__(self, ha, hylaa_settings):

        assert isinstance(hylaa_settings, HylaaSettings)
        assert isinstance(ha, LinearHybridAutomaton)

        if not openblas.has_openblas():
            print("Performance warning: OpenBLAS not detected. Matrix operations may be slower than necessary.")
            print("Is numpy linked with OpenBLAS? (hylaa.operblas.has_openblas() returned False)")

        if hylaa_settings.simulation.threads is not None:
            openblas.set_num_threads(hylaa_settings.simulation.threads)

        self.hybrid_automaton = ha
        self.settings = hylaa_settings
        self.num_vars = len(ha.variables)

        if self.settings.plot.plot_mode != PlotSettings.PLOT_NONE:
            Star.init_plot_vecs(self.num_vars, self.settings.plot)

        self.plotman = PlotManager(self, self.settings.plot)

        # computation
        self.waiting_list = WaitingList()
        self.error_stars_list = WaitingList()
        # self.reachable_stars_list = WaitingList()
        self.reach_tree = ReachTree()

        self.cur_state = None  # a Star object
        self.cur_step_in_mode = None  # how much dwell time in current continuous post
        self.max_steps_remaining = None  # bound on num steps left in current mode ; assigned on pop
        self.cur_sim_bundle = None  # set on pop
        self.discrete_dyn = hylaa_settings.discrete_dyn

        self.reached_error = False
        self.result = None  # a HylaaResult... assigned on run()

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            self.settings.simulation.use_presimulation = True

    def load_waiting_list(self, init_list):
        '''convert the init list into self.waiting_list'''

        assert len(init_list) > 0, "initial list length is 0"

        for mode, shape in init_list:
            assert isinstance(mode, LinearAutomatonMode)

            if isinstance(shape, HyperRectangle):
                star = init_hr_to_star(self.settings, shape, mode)
            elif isinstance(shape, list):
                assert len(shape) > 0, "initial constraints in mode '{}' was empty list".format(mode.name)
                assert isinstance(shape[0], LinearConstraint)

                star = init_constraints_to_star(self.settings, shape, mode)
            else:
                raise RuntimeError("Unsupported initial state type '{}': {}".format(type(shape), shape))

            self.waiting_list.add_deaggregated(star)

    def is_finished(self):
        'is the computation finished'

        rv = self.waiting_list.is_empty() and self.cur_state is None

        return rv or (self.settings.stop_when_error_reachable and self.reached_error)

    def check_guards(self, state, node):
        'check for discrete successors with the guards'

        assert state is not None

        for i in range(len(state.mode.transitions)):
            lp_solution = state.get_guard_intersection(i)

            if lp_solution is not None:
                transition = state.mode.transitions[i]
                successor_mode = transition.to_mode

                if successor_mode.is_error:
                    self.reached_error = True
                    node.error = True

                    if self.settings.stop_when_error_reachable:
                        raise FoundErrorTrajectory("Found error trajectory")

                # copy the current star to be the frozen pre-state of the discrete post operation
                discrete_prestate_star = state.clone()
                discrete_prestate_star.parent = ContinuousPostParent(state.mode, state.parent.star)

                # if successor_mode.is_error:
                #    self.error_stars_list.add_deaggregated(discrete_prestate_star)
                #    return

                discrete_poststate_star = state.clone()
                discrete_poststate_star.fast_forward_steps = 0  # reset fast forward on transitions
                basis_center = state.vector_to_star_basis(state.center)

                discrete_poststate_star.parent = DiscretePostParent(state.mode, discrete_prestate_star,
                                                                    basis_center, transition)

                # add each of the guard conditions to discrete_poststate_star
                for lin_con in transition.condition_list:

                    # first, convert the condition to the star's basis

                    # basis vectors (non-transpose) * standard_condition
                    basis_influence = np.dot(state.basis_matrix, lin_con.vector)
                    center_value = np.dot(state.center, lin_con.vector)
                    remaining_value = lin_con.value - center_value

                    basis_lc = LinearConstraint(basis_influence, remaining_value)
                    discrete_poststate_star.add_basis_constraint(basis_lc)

                if transition.reset_matrix is not None:
                    raise RuntimeError("only empty resets are currently supported")

                # there may be minor errors when converting the guard to the star's basis, so
                # re-check for feasibility

                if discrete_poststate_star.is_feasible():
                    violation_basis_vec = lp_solution[state.num_dims:]

                    if not self.settings.aggregation or not self.settings.deaggregation or \
                       self.has_counterexample(state, violation_basis_vec, state.total_steps):

                        # convert origin offset to star basis and add to basis_center
                        successor = discrete_poststate_star
                        # successor.start_center = successor.center

                        # Convert the basis_center into constraints to keep the center as zero for further simuation
                        successor.center_into_constraints(basis_center)

                        # self.plotman.cache_star_verts(successor) # do this before committing temp constriants

                        successor.start_basis_matrix = state.basis_matrix
                        # successor.basis_matrix = None # gets assigned from sim_bundle on pop

                        successor.mode = transition.to_mode

                        if successor_mode.is_error:
                            self.error_stars_list.add_deaggregated(discrete_prestate_star)
                        elif self.settings.aggregation:
                            self.waiting_list.add_aggregated(successor, self.settings)
                        else:
                            self.waiting_list.add_deaggregated(successor)
                            discrete_successor = successor
                            discrete_succ_node = self.reach_tree.add_node(discrete_successor, 1)
                            node.new_transition(discrete_succ_node)
                    else:
                        # a refinement occurred, stop processing guards
                        self.cur_state = state = None
                        break

    def deaggregate_star(self, star, steps_in_cur_star):
        'split an aggregated star in half, and place each half into the waiting list'

        Timers.tic('deaggregate star')

        assert isinstance(star.parent, AggregationParent)

        elapsed_aggregated_steps = steps_in_cur_star - star.total_steps - 1

        if elapsed_aggregated_steps < 0:  # happens on urgent transitions
            elapsed_aggregated_steps = 0

        mode = star.parent.mode
        all_stars = star.parent.stars

        # fast forward stars
        for s in all_stars:
            s.total_steps += elapsed_aggregated_steps
            s.fast_forward_steps += elapsed_aggregated_steps

        mid = len(all_stars) / 2
        left_stars = all_stars[:mid]
        right_stars = all_stars[mid:]

        for stars in [left_stars, right_stars]:
            discrete_post_star = stars[0]
            assert isinstance(discrete_post_star.parent, DiscretePostParent)
            discrete_pre_star = discrete_post_star.parent.prestar
            assert isinstance(discrete_pre_star.parent, ContinuousPostParent)

            if len(stars) == 1:
                # this might be parent of parent
                cur_star = discrete_post_star
            else:
                cur_star = make_aggregated_star(stars, self.settings)

            cur_star.mode = mode
            self.waiting_list.add_deaggregated(cur_star)

        Timers.toc('deaggregate star')

    def has_counterexample(self, star, basis_point, steps_in_cur_star):
        'check if the given basis point in the given star corresponds to a real trace'

        # if the parent is an initial state, then we're done and plot
        if isinstance(star.parent, InitParent):
            rv = True
        elif isinstance(star.parent, ContinuousPostParent):
            rv = self.has_counterexample(star.parent.star, basis_point, steps_in_cur_star)

            if not rv:
                # rv was false, some refinement occurred and we need to delete this star

                print ("; make this a setting, deleting aggregated from plot")
                # self.plotman.del_parent_successors(star.parent)
        elif isinstance(star.parent, DiscretePostParent):
            # we need to modify basis_point based on the parent's center
            basis_point = basis_point - star.parent.prestar_basis_center

            rv = self.has_counterexample(star.parent.prestar, basis_point, steps_in_cur_star)
        elif isinstance(star.parent, AggregationParent):
            # we need to SPLIT this aggregation parent
            rv = False

            self.deaggregate_star(star, steps_in_cur_star)
        else:
            raise RuntimeError("Concrete trace for parent type '{}' not implemented.".format(type(star.parent)))

        if rv and self.plotman.settings.plot_mode != PlotSettings.PLOT_NONE:
            if isinstance(star.parent, ContinuousPostParent):
                sim_bundle = star.parent.mode.get_existing_sim_bundle()
                num_steps = star.fast_forward_steps + star.total_steps - star.parent.star.total_steps
                start_basis_matrix = star.start_basis_matrix

                self.plotman.plot_trace(num_steps, sim_bundle, start_basis_matrix, basis_point)

        return rv

    def do_step_pop(self, output):
        'do a step where we pop from the waiting list'

        self.plotman.state_popped()  # reset certain per-mode plot variables

        self.cur_step_in_mode = 0

        if output:
            self.waiting_list.print_stats()

        parent_star = self.waiting_list.pop()

        if output:
            print("Removed state in mode '{}' at time {:.2f}; fast_forward_steps = {}".format(
                parent_star.mode.name, parent_star.total_steps * self.settings.step, parent_star.fast_forward_steps))

        self.max_steps_remaining = self.settings.num_steps - parent_star.total_steps + parent_star.fast_forward_steps
        self.cur_sim_bundle = parent_star.mode.get_sim_bundle(self.settings, parent_star, self.max_steps_remaining)

        state = parent_star.clone()

        state.parent = ContinuousPostParent(state.mode, parent_star)
        self.cur_state = state

        if self.settings.process_urgent_guards:
            self.check_guards(self.cur_state)

            if self.cur_state is None:
                if output:
                    print("After urgent checking guards, state was refined away.")
            elif output:
                print("Doing continuous post in mode '{}': ".format(self.cur_state.mode.name))

        if self.cur_state is not None and state.mode.is_error:
            self.cur_state = None

            if output:
                print ("Mode '{}' was an error mode; skipping.".format(state.mode.name))

        # pause after discrete post when using PLOT_INTERACTIVE
        if self.plotman.settings.plot_mode == PlotSettings.PLOT_INTERACTIVE:
            self.plotman.interactive.paused = True

    def do_step_continuous_post(self, output):
        '''do a step where it's part of a continuous post'''

        # advance current state by one time step
        state = self.cur_state

        if self.cur_step_in_mode == 0:
            current_node = self.reach_tree.get_node(state, 1)
        else:
            current_node = self.reach_tree.get_node(state, 0)

        if state.total_steps >= self.settings.num_steps:
            self.cur_state = None
        else:
            sim_bundle = self.cur_sim_bundle

            if self.settings.print_output and not self.settings.skip_step_times:
                print("Step: {} / {} ({})".format(self.cur_step_in_mode + 1, self.settings.num_steps,
                                                  self.settings.step * self.cur_step_in_mode))

            sim_step = self.cur_step_in_mode + 1 + state.fast_forward_steps

            new_basis_matrix, new_center = sim_bundle.get_vecs_origin_at_step(sim_step, self.max_steps_remaining, self.discrete_dyn)
            # print(new_basis_matrix, new_center)

            # if False:
            #     if self.discrete_dyn is True:
            #         f_name = 'basis_center_disc'
            #     else:
            #         f_name = 'basis_center_cont'
            #
            #     # if path.exists(f_name):
            #     #    os.remove(f_name)
            #     vals_f = open(f_name, "a")
            #     vals_f.write(str(new_basis_matrix))
            #     vals_f.write("\n")
            #     vals_f.write(str(new_center))
            #     vals_f.write("\n")
            #     vals_f.close()

            state.update_from_sim(new_basis_matrix, new_center)

            # increment step
            self.cur_step_in_mode += 1
            state.total_steps += 1

            continuous_successor_state = self.cur_state.clone()
            continuous_succ_node = self.reach_tree.add_node(continuous_successor_state, 0)
            current_node.new_transition(continuous_succ_node)

            self.check_guards(self.cur_state, continuous_succ_node)
            # self.reachable_stars_list.add_deaggregated(self.cur_state.clone())

            # refinement may occur while checking guards, which sets cur_state to None
            if self.cur_state is None:
                if output:
                    print("After checking guards, state was refined away.")
            else:
                is_still_feasible, inv_vio_star_list = self.cur_state.trim_to_invariant()

                for star in inv_vio_star_list:
                    self.plotman.add_inv_violation_star(star)

                if not is_still_feasible:
                    self.cur_state = None

        # after continuous post completes
        if self.cur_state is None:
            if self.plotman.settings.plot_mode == PlotSettings.PLOT_INTERACTIVE:
                self.plotman.interactive.paused = True

    def do_step(self):
        'do a single step of the computation'

        skipped_plot = False  # if we skip the plot, do multiple steps

        while True:
            output = self.settings.print_output
            self.plotman.reset_temp_polys()

            if self.cur_state is None:
                self.do_step_pop(output)
            else:
                try:
                    self.do_step_continuous_post(output)
                except FoundErrorTrajectory:  # this gets raised if an error mode is reachable and we should quit early
                    pass

            if self.cur_state is not None:
                skipped_plot = self.plotman.plot_current_star(self.cur_state)

            if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE or not skipped_plot or self.is_finished():
                break

        if self.is_finished() and self.settings.print_output:
            # result_f = open('./result.txt', 'a')
            if self.reached_error:
                print("Result: Error modes are reachable.\n")
                # result_f.write("Result: Error modes are reachable.\n")
            else:
                print("Result: Error modes are NOT reachable.\n")
                # result_f.write("Result: Error modes are not reachable.\n")

    def run_to_completion(self):
        'run the computation until it finishes (without plotting)'

        Timers.tic("total")

        while not self.is_finished():
            self.do_step()

        Timers.toc("total")

        if self.settings.print_output:
            LpInstance.print_stats()
            # Timers.print_stats()

        self.result.time = Timers.timers["total"].total_secs

    def run(self, init_list):
        '''
        run the computation

        init is a list of (LinearAutomatonMode, HyperRectangle)
        '''

        assert len(init_list) > 0

        # strengthen guards to include invariants of targets
        ha = init_list[0][0].parent
        ha.do_guard_strengthening()

        self.result = HylaaResult()
        self.plotman.create_plot()

        # convert init states to stars
        self.load_waiting_list(init_list)

        if self.settings.plot.plot_mode == PlotSettings.PLOT_NONE:
            # run without plotting
            self.run_to_completion()
        elif self.settings.plot.plot_mode == PlotSettings.PLOT_MATLAB:
            # matlab plot
            self.run_to_completion()
            self.plotman.save_matlab()
        else:
            # plot during computation
            self.plotman.compute_and_animate(self.do_step, self.is_finished)
        print("Waiting list '{}'".format((self.waiting_list.print_stats())))

        # error_stars = []
        # while not self.error_stars_list.is_empty():
        #     error_star = self.error_stars_list.pop()
        #     error_stars.append(error_star)

        return self.reach_tree


class WaitingList(object):
    '''
    The set of to-be computed values (discrete sucessors).

    There are aggregated states, and deaggregated states. The idea is states first
    go into the aggregrated ones, but may be later split and placed into the
    deaggregated list. Thus, deaggregated states, if they exist, are popped first.
    The states here are Star instances
    '''

    def __init__(self):
        self.aggregated_mode_to_state = OrderedDict()
        self.deaggregated_list = []

    def pop(self):
        'pop a state from the waiting list'

        assert len(self.aggregated_mode_to_state) + len(self.deaggregated_list) > 0, \
            "pop() called on empty waiting list"

        if len(self.deaggregated_list) > 0:
            rv = self.deaggregated_list[0]
            self.deaggregated_list = self.deaggregated_list[1:]
        else:
            # pop from aggregated list
            rv = self.aggregated_mode_to_state.popitem(last=False)[1]

            assert isinstance(rv, Star)

        return rv

    def print_stats(self):
        'print statistics about the waiting list'

        total = len(self.aggregated_mode_to_state) + len(self.deaggregated_list)

        print("Waiting list contains {} states ({} aggregated and {} deaggregated):".format(
            total, len(self.aggregated_mode_to_state), len(self.deaggregated_list)))

        counter = 1

        for star in self.deaggregated_list:
            print(" {}. Deaggregated Successor in Mode '{}'".format(counter, star.mode.name))
            counter += 1

        for mode, star in self.aggregated_mode_to_state.items():
            if isinstance(star.parent, AggregationParent):
                print(" {}. Aggregated Sucessor in Mode '{}': {} stars".format(counter, mode, len(star.parent.stars)))
            else:
                # should be a DiscretePostParent
                print(" {}. Aggregated Sucessor in Mode '{}': single star".format(counter, mode))

            counter += 1

    def is_empty(self):
        'is the waiting list empty'

        return len(self.deaggregated_list) == 0 and len(self.aggregated_mode_to_state) == 0

    def add_deaggregated(self, state):
        'add a state to the deaggregated list'

        assert isinstance(state, Star)

        self.deaggregated_list.append(state)

    def add_aggregated(self, new_star, hylaa_settings):
        'add a state to the aggregated map'

        assert isinstance(new_star, Star)
        assert new_star.basis_matrix is not None

        mode_name = new_star.mode.name

        existing_state = self.aggregated_mode_to_state.get(mode_name)

        if existing_state is None:
            self.aggregated_mode_to_state[mode_name] = new_star
        else:
            # combine the two stars
            cur_star = existing_state

            cur_star.total_steps = min(cur_star.total_steps, new_star.total_steps)

            # if the parent of this star is not an aggregation, we need to create one
            # otherwise, we need to add it to the list of parents

            if isinstance(cur_star.parent, AggregationParent):
                # parent is already an aggregation. add it to the list of parents and eat it
                cur_star.parent.stars.append(new_star)
                cur_star.eat_star(new_star)
            else:
                # create the aggregation parent
                hull_star = make_aggregated_star([cur_star, new_star], hylaa_settings)

                self.aggregated_mode_to_state[mode_name] = hull_star


class FoundErrorTrajectory(RuntimeError):
    'gets thrown if a trajectory to the error states is found when settings.stop_when_error_reachable is True'
