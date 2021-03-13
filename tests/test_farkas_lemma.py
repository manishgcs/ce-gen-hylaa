from farkas_central.farkas_mip import FarkasMIP
from farkas_central.farkas_smt import FarkasSMT
from hylaa.polytope import create_polytope_4m_intervals

if __name__ == '__main__':
    # p1_intervals = [[1, 9], [0, 6]]
    # p2_intervals = [[2, 5], [-2, 4]]
    # q_set_intervals = [[[4, 8], [-1, 2]], [[5, 10], [5, 7]], [[7, 11], [5, 8]]]
    # q_set_intervals = [[[4, 8], [4, 6]], [[5, 10], [5, 7]], [[7, 11], [5, 8]]]
    # q_set_intervals = [[[4, 8], [-1, 2]], [[5, 10], [5, 7]], [[4, 7], [0, 1]]]

    p1_intervals = [[2, 10], [2, 10]]
    p2_intervals = [[7, 14], [7, 12]]
    q_set_intervals = [[[1, 5], [0, 6]], [[4, 9], [4, 9]], [[11, 12], [0, 8]], [[3, 12], [3, 5]]]

    q_set = []
    n_dims = len(p1_intervals)
    for q_intervals in q_set_intervals:
        q_polytope = create_polytope_4m_intervals(q_intervals, n_dims)
        # q_poly.polytopePrint()
        q_set.append(q_polytope)
    p1 = create_polytope_4m_intervals(p1_intervals, n_dims)
    p2 = create_polytope_4m_intervals(p2_intervals, n_dims)
    test_instance_mip = FarkasMIP(p2, p1, q_set, n_dims)
    z_vals, alpha_vals = test_instance_mip.solve_4_both_mip()
    # test_instance_mip.solve_4_one_polytope_mip(poly_idx=2)
    print(z_vals, alpha_vals)

    test_instance_smt = FarkasSMT(p2, p1, q_set, n_dims)
    # test_instance_smt.solve_4_both_smt_file()
    z_vals, alpha_vals = test_instance_smt.solve_4_both_smt()
    print(z_vals, alpha_vals)
    # s = Optimize()
    # y_1 = Real('y_1')
    # c1 = y_1 > 10
    # c2 = And(c1, y_1 > 11)
    # s.add(c2)
    # print(s.check())
    # m = s.model()
    # print(m[y_1])
    # Timers.print_stats()

