import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from hypercube import generate_all_ising_states


def calc_energy(interaction_matrix, all_states, external_field=None):
    if external_field is None:
        external_field = np.zeros(all_states.shape[1])
    energy = []
    for i in range(len(all_states)):
        state = all_states[i]
        e = -0.5 * np.dot(state, np.dot(interaction_matrix, state)) - np.dot(
            state, external_field
        )
        energy.append(e)
    energy = np.array(energy)
    return energy


def calc_energy_and_canonical_probability(
    interaction_matrix, temperature, external_field=None, calc_probability=True
):
    all_states = generate_all_ising_states(interaction_matrix.shape[0])
    all_states = all_states.astype(np.float64)
    energy = calc_energy(interaction_matrix, all_states, external_field)
    if calc_probability:
        weight = np.exp(-energy / temperature)
        weight /= weight.sum()
        return energy, weight
    else:
        return energy


## kagome spin ice
def ferromagnetic_kagome(
    Lx,
    Ly,
    pbc=True,
    cyclic=False,
    halve_strength=False,
    triangle_shape=False,
    lower_left_corner=True,
    remove_corner="llur",
    return_coord_arrow=False,
):
    """
    This function can return interaction matrix, 2d-coordinates of spins
    and vector of arrow for spin ice visualization
    e.g.
    im, coord, arrow_vec = ims.antiferromagnetic_kagome(L)
    """
    # create square triangular interaction matrix
    im = np.zeros([Lx * Ly, Lx * Ly])
    for spin_id in range(Lx * Ly):
        i = spin_id // Lx
        j = spin_id - Lx * i
        # target0 right spin
        target0 = i * Lx + (j + 1) % Lx
        # target1 upper spin
        target1 = (i + 1) % Ly * Lx + j
        # target2 upper left spin
        target2 = (i + 1) % Ly * Lx + (j + Lx - 1) % Lx
        im[target0, spin_id] = 1
        im[target1, spin_id] = 1
        im[target2, spin_id] = 1
    if not pbc:
        """
        remove starategy
        e.g. Lx=4, Ly=3

        upper boundary
        |_|_|_| <- upper interaction
        |_|_|_|
        |_|_|_|

        upper left boundary
        \_\_\_\  <- upper left interaction
        |_|_|_|
        |_|_|_|

        right boundary
         _ _ _ _
        |_|_|_|_
        |_|_|_|_
               ^
               |
               right interaction

        left boundary
        \ _ _ _
        \|_|_|_|
        \|_|_|_|
        ^
        |
        upper left interaction

        """
        # remove interaction from up edge
        i = Ly - 1
        # delete upper boudary
        for j in range(Lx):
            spin_id = i * Lx + j
            target = j
            im[target, spin_id] = 0
        # delete upper left boudary
        for j in range(Lx):
            spin_id = i * Lx + j
            target = j - 1
            im[target, spin_id] = 0
        # remove interaction from right edge
        j = Lx - 1
        # delete right boudary
        for i in range(Ly):
            spin_id = i * Lx + j
            target = i * Lx
            im[target, spin_id] = 0
        # delete upper left boudary
        for i in range(Ly):
            spin_id = i * Lx
            target = (i + 1) % Ly * Lx + j
            im[target, spin_id] = 0
    # make interaction matrix symmetric
    if not cyclic:
        im = im + im.T
    if halve_strength:
        im *= 0.5

    # create square triangular coordinates
    unit_vec0 = np.array([1 / 2, np.sqrt(3) / 2])
    unit_vec1 = np.array([1, 0])
    i = np.arange(Lx * Ly) // Lx
    j = np.arange(Lx * Ly) - Lx * i
    x = i * unit_vec0[0] + j * unit_vec1[0]
    y = i * unit_vec0[1] + j * unit_vec1[1]
    coord = np.stack([x, y], axis=1)

    # create a vector for each spin
    shift0 = np.array([1 / 2, 1 / (2 * np.sqrt(3))])
    shift1 = np.array([1, 1 / np.sqrt(3)])
    ti = np.arange(Lx * Ly) // Lx // 2
    tj = (np.arange(Lx * Ly) - Lx * i) // 2
    if lower_left_corner:
        shift = shift0
    else:
        shift = shift1
    tx = ti * 2 * unit_vec0[0] + tj * 2 * unit_vec1[0] + shift[0]
    ty = ti * 2 * unit_vec0[1] + tj * 2 * unit_vec1[1] + shift[1]
    tcoord = np.stack([tx, ty], axis=1)
    if lower_left_corner:
        arrow_vec = tcoord - coord
    else:
        arrow_vec = -(tcoord - coord)

    # create remove list
    remove_list = []
    # for triangle shape
    if triangle_shape:
        for i in range(Ly - 1, 0, -1):
            for j in range(Lx - 1, Lx - 1 - i, -1):
                remove_list.append(i * Lx + j)

    # by removing nodes of triangular lattice periodically, we obtain kagome lattice
    if lower_left_corner:
        for i in range(1, Ly, 2):
            for j in range(1, Lx, 2):
                target = i * Lx + j
                if target not in remove_list:
                    remove_list.append(target)
    else:
        for i in range(0, Ly, 2):
            for j in range(0, Lx, 2):
                target = i * Lx + j
                if target not in remove_list:
                    remove_list.append(target)

    # removing all corner nodes of triangular lattice
    if remove_corner == "all":
        for i in [0, Ly - 1]:
            for j in [0, Lx - 1]:
                target = i * Lx + j
                if target not in remove_list:
                    remove_list.append(target)
    # removing lower left and upper right (llur) corner nodes
    elif remove_corner == "llur":
        for i, j in zip([0, Ly - 1], [0, Lx - 1]):
            target = i * Lx + j
            if target not in remove_list:
                remove_list.append(target)

    # remove node according to remove list
    im = np.delete(im, remove_list, axis=0)
    im = np.delete(im, remove_list, axis=1)
    coord = np.delete(coord, remove_list, axis=0)
    arrow_vec = np.delete(arrow_vec, remove_list, axis=0)

    # centralize the coordinates, normalize arrow vector
    coord[:, 0] = coord[:, 0] - np.mean(coord[:, 0])
    coord[:, 1] = coord[:, 1] - np.mean(coord[:, 1])
    arrow_vec = arrow_vec / np.linalg.norm(arrow_vec, axis=1, keepdims=True)

    if return_coord_arrow:
        return im, coord, arrow_vec
    else:
        return im


def antiferromagnetic_kagome(
    Lx,
    Ly,
    pbc=True,
    cyclic=False,
    halve_strength=False,
    triangle_shape=False,
    lower_left_corner=True,
    remove_corner="llur",
    return_coord_arrow=False,
):
    im, coord, arrow_vec = ferromagnetic_kagome(
        Lx,
        Ly,
        pbc=pbc,
        cyclic=cyclic,
        halve_strength=halve_strength,
        triangle_shape=triangle_shape,
        lower_left_corner=lower_left_corner,
        remove_corner=remove_corner,
        return_coord_arrow=True,
    )
    im *= -1

    if return_coord_arrow:
        return im, coord, arrow_vec
    else:
        return im


def Farhan2013(num_of_ring=3, return_coord_arrow=True):
    """
    This function creates the ring of kagome lattice.
    See details at
    https://doi.org/10.1038/nphys2613
    """
    if num_of_ring == 1:
        Lx = 3
        Ly = 3
        pbc = False
        triangle_shape = False
        remove_corner = "llur"
    elif num_of_ring == 2:
        Lx = 5
        Ly = 3
        pbc = False
        triangle_shape = False
        remove_corner = "llur"
    elif num_of_ring == 3:
        Lx = 6
        Ly = 6
        pbc = False
        triangle_shape = True
        remove_corner = "all"
    else:
        raise ValueError("num_of_ring must be one of three: 1, 2, or 3")

    return antiferromagnetic_kagome(
        Lx,
        Ly,
        pbc=pbc,
        triangle_shape=triangle_shape,
        remove_corner=remove_corner,
        return_coord_arrow=return_coord_arrow,
    )


def ferromagnetic_all2all(N):
    im = np.ones((N, N))
    np.fill_diagonal(im, 0)
    return im


def _grid_1d(strength, N, pbc=True):
    """
    pbc stands for periodic boundary condition
    """
    im = np.zeros((N, N))
    for i in range(N):
        im[(i + 1) % N, i] = strength

    if not pbc:
        im[0, N - 1] = 0
    im = im + im.T
    return im


def _grid_2d(strength, L, pbc=True):
    """
    pbc stands for periodic boundary condition
    """
    N = L**2
    im = np.zeros((N, N))
    for x in range(L):
        for y in range(L):
            # source
            j = L**0 * x + L**1 * y
            # to the right
            i = L**0 * ((x + 1) % L) + L**1 * y
            im[i, j] = strength
            # to the bottom
            i = L**0 * x + L**1 * ((y + 1) % L)
            im[i, j] = strength

    if not pbc:
        x = L - 1
        for y in range(L):
            # source
            j = L**0 * x + L**1 * y
            # to the bottom
            i = L**0 * ((x + 1) % L) + L**1 * y
            im[i, j] = 0
        y = L - 1
        for x in range(L):
            # source
            j = L**0 * x + L**1 * y
            # to the bottom
            i = L**0 * x + L**1 * ((y + 1) % L)
            im[i, j] = 0
    im = im + im.T
    return im


def ferromagnetic_1d(N, pbc=True):
    """
    pbc stands for periodic boundary condition
    """
    return _grid_1d(1.0, N, pbc)


def ferromagnetic_2d(L, pbc=True):
    return _grid_2d(1.0, L, pbc)


def sylvester_hadamard_matrix(k):
    k1 = np.array(
        [
            [+1, +1],
            [+1, -1],
        ]
    )
    matrix = k1.copy()
    for _ in range(k - 1):
        matrix = np.kron(k1, matrix)
    return matrix


def sh_matrix(k):
    return sylvester_hadamard_matrix(k)


def plot_kagome_spin_ice(
    coord,
    arrow,
    states1d=None,
    cmap=cc.cm.gwv_r,
    width=0.1,
    margins=0.3,
    return_xyq=False,
    ax=None,
    figsize=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if states1d is None:
        states1d = np.ones(len(coord))

    x = coord[:, 0] - np.mean(coord[:, 0])
    y = coord[:, 1] - np.mean(coord[:, 1])
    u = arrow[:, 0] * states1d
    v = arrow[:, 1] * states1d
    c = np.round(x * v - y * u, 2)
    q = ax.quiver(
        x,
        y,
        u,
        v,
        np.sign(c),
        cmap=cmap,
        clim=(-1.1, 1.1),
        pivot="mid",
        scale=1.0,
        scale_units="xy",
        units="xy",
        width=width,
        lw=0.01,
        edgecolor="k",
    )

    ax.set_aspect("equal")
    ax.margins(margins)
    ax.axis("off")

    if return_xyq:
        return ax, x, y, q
    else:
        return ax


def _plot_network(
    im,
    coord,
    node_color=None,
    labels=None,
    cmap=plt.cm.binary,
    vmin=None,
    vmax=None,
    node_size=100,
    linewidths=0.1,
    edge_cmap=cc.cm.CET_D3_r,
    edge_width_scale=1.0,
    font_size=7,
    font_color="w",
    font_family="sans-serif",
    font_weight="normal",
    ax=None,
    figsize=None,
    dpi=None,
):
    G = nx.from_numpy_array(im, create_using=nx.Graph())
    if isinstance(coord, (list, np.ndarray)):
        pos = {i: coord[i] for i in range(len(coord))}
    elif coord == "circular":
        pos = nx.circular_layout(G)
    else:
        raise ValueError("coord must be list, np.ndarray, or 'circular'")
    if node_color is None:
        node_color = np.ones(len(pos))

    if vmin is None and vmax is None:
        vabsmax = 1
        vmin = -vabsmax
        vmax = vabsmax

    edge_weights = [edgedata["weight"] for _, _, edgedata in G.edges(data=True)]
    edge_width = np.abs(edge_weights) * edge_width_scale
    edge_sign = np.sign(edge_weights)
    if edge_cmap is None:
        edge_color = "k"
    else:
        edge_color = edge_sign.copy()
    edge_style = edge_sign.copy().astype(str)
    edge_style = np.where(edge_style == "1.0", "-", edge_style)
    edge_style = np.where(edge_style == "0.0", "-", edge_style)
    edge_style = np.where(edge_style == "-1.0", "--", edge_style)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        alpha=1.0,
        node_size=node_size,
        node_color=node_color,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        edgecolors="black",
        linewidths=linewidths,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        node_size=node_size,
        arrows=False,
        style=edge_style,
        width=edge_width,
        edge_color=edge_color,
        edge_cmap=edge_cmap,
        edge_vmin=-1.1,
        edge_vmax=1.1,
    )
    if labels is not None:
        if labels == "auto":
            labels = {i: i + 1 for i in np.arange(len(pos))}
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            labels=labels,
            font_size=font_size,
            font_color=font_color,
            font_family=font_family,
            font_weight=font_weight,
        )

    ax.set_aspect("equal")
    ax.axis("off")

    if ax is None:
        return fig, ax
    else:
        return ax


def plot_kagome_network(
    im,
    coord,
    node_color=None,
    labels=None,
    cmap=plt.cm.binary,
    vmin=None,
    vmax=None,
    node_size=100,
    linewidths=0.1,
    edge_cmap=cc.cm.CET_D3_r,
    edge_width_scale=1.0,
    font_size=7,
    font_color="w",
    font_family="sans-serif",
    font_weight="normal",
    ax=None,
    figsize=None,
    dpi=None,
):
    return _plot_network(
        im,
        coord,
        node_color,
        labels,
        cmap,
        vmin,
        vmax,
        node_size,
        linewidths,
        edge_cmap,
        edge_width_scale,
        font_size,
        font_color,
        font_family,
        font_weight,
        ax,
        figsize,
        dpi,
    )


def plot_network_circulary(
    im,
    node_color=None,
    labels=None,
    cmap=plt.cm.binary,
    vmin=None,
    vmax=None,
    node_size=100,
    linewidths=0.1,
    edge_cmap=cc.cm.CET_D3_r,
    arrowsize=10,
    arrow_width_scale=1.0,
    arrow_rad=0.0,
    font_size=7,
    font_color="w",
    font_family="sans-serif",
    font_weight="normal",
    ax=None,
    figsize=None,
    dpi=None,
):
    return _plot_network(
        im,
        "circular",
        node_color,
        labels,
        cmap,
        vmin,
        vmax,
        node_size,
        linewidths,
        edge_cmap,
        arrowsize,
        arrow_width_scale,
        arrow_rad,
        font_size,
        font_color,
        font_family,
        font_weight,
        ax,
        figsize,
        dpi,
    )
