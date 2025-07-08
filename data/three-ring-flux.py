import sys

import numpy as np

sys.path.append("src")
import dynamics as dn
import networks as ns

args = sys.argv
with_external_field = int(args[1])


def bin2int(array):
    return int("".join(map(str, array)), 2)


# T = 0.3
T = 8.0
im, coord, arrow = ns.Farhan2013(3)
if with_external_field:
    external_field = np.array([1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1])
    initial_probability_vector = np.zeros(2 ** len(im))
    initial_probability_vector[bin2int(((-external_field + 1) / 2).astype(int))] = 1.0
    results = dn.calc_time_evolution(
        im,
        T,
        external_field,
        initial_probability_vector=initial_probability_vector,
        final_time=50,
    )
else:
    results = dn.calc_time_evolution(im, T)
flux_edge_list = results.flux_edge_list

file_name = f"data/flux_edge_list_3_ring_T={T:.1f}"
if with_external_field:
    file_name += "_field"
np.savez(
    file_name,
    source_list=flux_edge_list[0],
    target_list=flux_edge_list[1],
    rate_list=flux_edge_list[2],
)
np.savez(
    file_name + "_probability_time",
    probability=results.probability_vectors,
    time=results.time,
)
