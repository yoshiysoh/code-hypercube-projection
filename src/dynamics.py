import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_array


def _arrhenius_weight(beta, energy1d, freq=1, barrier=0):
    weight = freq * np.exp(beta * (energy1d - barrier))
    return weight


def _glauber_weight(beta, energy1d, freq=1, barrier=0):
    # weight = np.exp(beta*2*energy1d) / (1+np.exp(beta*2*energy1d))
    weight = 1.0 / (1.0 + np.exp(-beta * 2 * energy1d))
    return weight


def temperature_to_beta(T):
    with np.errstate(divide="ignore"):
        beta = np.divide(1.0, T)
    beta = np.nan_to_num(beta)
    return beta


def int2binary(number, width):
    """
    Encode numbers as binary arrays of given width.
    """
    code = [(number >> i) & 1 for i in range(width)]
    code = np.array(code)[::-1]
    return code


def binary_to_int(binary):
    if isinstance(binary, (list, np.ndarray)):
        binary = "".join(str(b) for b in binary)
    integer = int("0b" + binary, 2)
    return integer


def generate_all_binary_state(Ndims):
    total = 2**Ndims
    states = np.zeros((total, Ndims))
    for i in range(total):
        state = int2binary(i, Ndims)
        states[i] = state
    return states


def generate_all_ising_state(Ndims):
    states = generate_all_binary_state(Ndims)
    states = 2.0 * states - 1.0
    return states


def _all_states_to_all_energy(all_states, im, external_field=None):
    if external_field is None:
        external_field = np.zeros(len(all_states[0]))
    energy = np.zeros((len(all_states), len(im)))
    for i in range(len(all_states)):
        state = all_states[i]
        input_field = im @ state
        energy1d = -state * input_field - state * external_field
        energy[i] = energy1d
    return energy


def _all_states_to_transition_rate(beta, all_states, energy, _weight_func):
    num_states = len(all_states)
    transition_rate = np.zeros((num_states, num_states))
    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                continue
            else:
                change = np.argwhere(all_states[i] * all_states[j] == -1)
                if len(change) != 1:
                    continue
                else:
                    transition_rate[i, j] = _weight_func(
                        beta, energy[j, change[0, 0]], freq=1, barrier=0
                    )

    for i in range(num_states):
        transition_rate[i, i] = -np.sum(transition_rate[:, i])
    transition_rate = csr_array(transition_rate)
    return transition_rate


def im_to_transition_rate(im, T, external_field=None, weight_type="arrhenius"):
    if weight_type == "arrhenius":
        _weight_func = _arrhenius_weight
    elif weight_type == "glauber":
        _weight_func = _glauber_weight
    else:
        raise ValueError("weight_type must be 'arrhenius' or 'glauber'")

    ## generate all state
    im = np.array(im).astype(float)
    Ndims = len(im)
    all_states = generate_all_ising_state(Ndims)

    energy = _all_states_to_all_energy(all_states, im, external_field)

    beta = temperature_to_beta(T)
    transition_rate = _all_states_to_transition_rate(
        beta, all_states, energy, _weight_func
    )

    return transition_rate


def transition_rate_to_time_evolution(
    transition_rate,
    initial_probability_vector=None,
    final_time=100.0,
    print_messeage=True,
):
    if initial_probability_vector is None:
        # initial_probability_vector = np.ones(len(transition_rate))
        initial_probability_vector = np.ones(transition_rate.shape[0])
    initial_probability_vector = (
        initial_probability_vector / initial_probability_vector.sum()
    )
    t_span = [0, final_time]

    def master_equation(t, probability_vector):
        return transition_rate @ probability_vector

    sol = solve_ivp(
        master_equation,
        t_span,
        initial_probability_vector,
        method="LSODA",
        # t_eval=t_eval,
        dense_output=True,
        vectorized=True,
        rtol=1e-8,
    )
    if print_messeage:
        print(sol.message)
    return sol


def im_to_time_evolution(
    im,
    T,
    external_field=None,
    initial_probability_vector=None,
    final_time=50.0,
    print_messeage=True,
    return_transition_rate=False,
):
    # creating transition_rate matrix from interaction matrix
    transition_rate = im_to_transition_rate(im, T, external_field)

    # solving master equation as a initial value problem
    sol = transition_rate_to_time_evolution(
        transition_rate,
        initial_probability_vector=initial_probability_vector,
        final_time=final_time,
        print_messeage=print_messeage,
    )
    if return_transition_rate:
        return sol, transition_rate
    else:
        return sol


def _nonzero_indexes_to_list(matrix, all_states, nonzero_indexes, num_spins):
    num_rates = len(nonzero_indexes[0])
    source_list = np.zeros((num_rates, num_spins))
    target_list = np.zeros((num_rates, num_spins))
    rate_list = np.zeros(num_rates)
    for i in range(num_rates):
        s = all_states[nonzero_indexes[1][i]]  # source state
        t = all_states[nonzero_indexes[0][i]]  # target state
        r = matrix[nonzero_indexes[0][i], nonzero_indexes[1][i]]  # rate
        source_list[i] = s
        target_list[i] = t
        rate_list[i] = r
    return source_list, target_list, rate_list


def flux_to_flux_list(flux):
    num_spins = np.log2(len(flux))
    num_spins = np.int64(num_spins)
    all_states = generate_all_ising_state(num_spins)
    nonzero_indexes = flux.nonzero()
    num_nonzero = len(nonzero_indexes[0])
    mask = np.zeros(num_nonzero, dtype=np.bool_)
    for n in range(num_nonzero):
        i = nonzero_indexes[0][n]
        j = nonzero_indexes[1][n]
        if flux[i, j] > 0.0:
            mask[n] = True
    nonzero_indexes = (nonzero_indexes[0][mask], nonzero_indexes[1][mask])
    return _nonzero_indexes_to_list(flux, all_states, nonzero_indexes, num_spins)


def calc_time_evolution(
    im,
    T,
    external_field=None,
    sol=None,
    transition_rate=None,
    initial_probability_vector=None,
    final_time=5.0,
    num_time_steps=10000,
    print_messeage=True,
):
    if sol is None or transition_rate is None:
        sol, transition_rate = im_to_time_evolution(
            im,
            T,
            external_field=external_field,
            initial_probability_vector=initial_probability_vector,
            final_time=final_time,
            print_messeage=print_messeage,
            return_transition_rate=True,
        )

    t_eval = np.linspace(sol.t[0], sol.t[-1], num_time_steps)
    time = t_eval
    probability_vectors = sol.sol(t_eval)

    # from transition_rate matrix and probability vectors, calculate time evolution of joint transition matrix
    # transition_rate_non_diagonal = transition_rate.copy()
    # np.fill_diagonal(transition_rate_non_diagonal, 0.0)
    # joint_transition_time_evolution = np.einsum(
    #    "ij, jt -> tij", transition_rate_non_diagonal, probability_vectors
    # )

    # calc flux rate from time evoution
    # flux_time_evolution = (
    #    joint_transition_time_evolution
    #    - joint_transition_time_evolution.transpose(0, 2, 1)
    # )
    # flux_sum = np.sum(flux_time_evolution, axis=0)
    # flux_time_integrated = flux_sum * np.diff(time)[0]
    # flux_edge_list = flux_to_flux_list(flux_time_integrated)

    transition_rate.setdiag(0.0)

    def calc_joint_transition(transition_rate, probability_vector):
        return transition_rate * np.tile(
            probability_vector, (len(probability_vector), 1)
        )

    def calc_flux(transition_rate, probability_vector):
        joint_transition = calc_joint_transition(transition_rate, probability_vector)
        return joint_transition - joint_transition.transpose()

    flux_sum = csr_array(
        (len(probability_vectors), len(probability_vectors)), dtype=np.float64
    )
    for i in range(probability_vectors.shape[1]):
        flux = calc_flux(transition_rate, probability_vectors[:, i])
        flux_sum += flux
    flux_sum = flux_sum.multiply(np.diff(time)[0])
    flux_edge_list = flux_to_flux_list(flux_sum.toarray())

    # class for result
    class result:
        def __init__(
            self,
            sol,
            transition_rate,
            time,
            probability_vectors,
            flux_edge_list,
        ):
            self.sol = sol
            self.transition_rate = transition_rate
            self.time = time
            self.probability_vectors = probability_vectors
            self.flux_edge_list = flux_edge_list

    return result(
        sol,
        transition_rate,
        time,
        probability_vectors,
        flux_edge_list,
    )
