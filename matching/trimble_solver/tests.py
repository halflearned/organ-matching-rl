from matching.trimble_solver.kidney_digraph import read_digraph
import matching.trimble_solver.kidney_ip as k_ip
import matching.trimble_solver.kidney_ndds as k_ndds
import matching.trimble_solver.kidney_utils as k_utils


def read_with_ndds(basename):
    with open(basename + ".input") as f:
        lines = f.readlines()
    d = read_digraph(lines)
    with open(basename + ".ndds") as f:
        lines = f.readlines()
    ndds = k_ndds.read_ndds(lines, d)
    return d, ndds

def solve(filepath, max_cycle, max_chain, formulation="hpief_prime_full_red"):

    fn = k_ip.optimise_hpief_prime_full_red
     #k_ip.optimise_hpief_prime,
     #k_ip.optimise_hpief_2prime,
     #k_ip.optimise_hpief_prime_full_red,
     #k_ip.optimise_hpief_2prime_full_red,
     #k_ip.optimise_picef, k_ip.optimise_ccf,

    d, ndds = read_with_ndds(filepath)
    opt_result = fn(k_ip.OptConfig(d, ndds, max_cycle, max_chain))
    return opt_result

