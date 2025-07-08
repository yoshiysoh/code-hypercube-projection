from itertools import combinations

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance


class WPCA:
    def fit(self, data, weights, symmetric_weights=True):
        if weights is None:
            weights = np.ones(len(data))
        p = weights / weights.sum()
        if symmetric_weights:
            # no need to extract mean because mean is zero when $p(\bm{s}) = p(-\bm{s})$
            # faster computation by avoiding centering
            cov = data.T * p @ data
        else:
            # centering data and calculating covariance
            p_vec = np.vstack(p)
            mean_data = np.mean(p_vec * data, axis=0)
            c_data = data - mean_data
            cov = c_data.T * p @ c_data
        values, vectors = np.linalg.eigh(cov)
        values = values[::-1]
        vectors = vectors.T[::-1]
        vectors = vectors * (2 * (vectors.sum(axis=1, keepdims=True) >= 0) - 1)
        self.components_ = vectors
        self.explained_variance_ = values
        self.explained_variance_ratio_ = values / values.sum()
        return self

    def transform(self, X):
        return X @ self.components_.T

    def fit_transform(self, X, weights, symmetric_weights):
        return self.fit(X, weights, symmetric_weights).transform(X)


def _int2bin(numbers, width):
    """
    Encode numbers as binary arrays of given width.
    """
    codes = []
    for number in numbers:
        code = [(number >> i) & 1 for i in reversed(range(width))]
        codes.append(code)
    return np.array(codes, order="C")


def _bin2int(binaries):
    base = 2 ** np.arange(len(binaries[0]))[::-1]
    return np.dot(binaries, base)


def _bin2labels(binaries):
    """
    Convert binary arrays to labels.
    """
    labels = ["".join(str(digit) for digit in code) for code in binaries]
    return np.array(labels)


def _bin2gray(binaries):
    """
    Convert binary arrays to Gray code.
    """
    grays = []
    for binary in binaries:
        gray = [binary[0]]
        for i in range(1, len(binary)):
            gray.append(binary[i] ^ binary[i - 1])
        grays.append(gray)
    return np.array(grays)


def generate_all_ising_states(num_dimensions):
    num_vertices = 2**num_dimensions
    vertices_ids = np.arange(num_vertices)
    vertex_binaries = _int2bin(vertices_ids, width=num_dimensions)
    vertex_ising = 2 * vertex_binaries - 1
    return vertex_ising


def calc_hypercube_coordinates(
    num_dimensions=4,
    style="isometric",
    distortion=0,
    fractal_repeat=2,
    fractal_length_ratio=10,
    pca_ground_state_distance=None,
    xcomponent=0,
    ycomponent=1,
    weights=None,
    symmetric_weights=True,
    flip_xbasis=False,
    flip_ybasis=False,
):
    vertex_ising = generate_all_ising_states(num_dimensions)
    vertex_binaries = (vertex_ising + 1) / 2
    vertex_binaries = vertex_binaries.astype(int)

    if style == "isometric":
        angles = np.pi * np.arange(num_dimensions) / num_dimensions
        angles += np.pi / (2 * num_dimensions)
        angles += np.linspace(0, distortion, len(angles))
        basis = np.transpose([np.cos(angles), np.sin(angles)])
    elif style == "hamming":
        y_max = num_dimensions // 2 + 1
        y_min = -num_dimensions // 2 + 1
        # y_basis = np.arange(y_min, y_max) - (num_dimensions + 1) % 2 * 0.5
        # y_basis += np.arange(len(y_basis)) % 2 * distortion
        y_unit = (y_max - y_min) / (num_dimensions - 1)
        y_basis = (np.arange(num_dimensions) + 1) - (num_dimensions + 1) / 2
        y_basis = y_basis * y_unit
        basis = np.transpose([np.ones(num_dimensions), y_basis])
    elif style == "fractal":
        if num_dimensions % fractal_repeat != 0:
            raise ValueError("num_dimensions must be divisible by fractal_repeat")
        effective_num_dimensions = int(num_dimensions // fractal_repeat)
        angles = np.pi * np.arange(effective_num_dimensions) / effective_num_dimensions
        angles += np.pi / (2 * effective_num_dimensions)
        angles = np.tile(angles, fractal_repeat)
        angles += np.linspace(0, distortion, len(angles))
        basis_length = np.ones(num_dimensions)
        basis_length[:effective_num_dimensions] *= fractal_length_ratio
        basis = np.transpose(
            [np.cos(angles) * basis_length, np.sin(angles) * basis_length]
        )
    # elif style == "pca" or style == "wpca":
    elif style == "wpca":
        if pca_ground_state_distance is not None:
            data = []
            data.append(np.ones(num_dimensions))
            data.append(-np.ones(num_dimensions))
            for i in range(pca_ground_state_distance):
                combs = list(combinations(range(num_dimensions), i + 1))
                for comb in combs:
                    for c in comb:
                        flipped_data = np.ones(num_dimensions)
                        flipped_data[c] *= -1
                        data.append(flipped_data)
                        data.append(-flipped_data)
            data = np.array(data)
        else:
            data = vertex_ising

        # if style == "pca":
        #    pca = PCA(svd_solver="full")
        #    pca.fit_transform(zscores)
        #    basis = np.transpose(
        #        [pca.components_[xcomponent], pca.components_[ycomponent]]
        #    )
        # elif style == "wpca":
        if style == "wpca":
            wpca = WPCA()
            wpca.fit_transform(
                data, weights=weights, symmetric_weights=symmetric_weights
            )
            basis = np.transpose(
                [wpca.components_[xcomponent], wpca.components_[ycomponent]]
            )

        if flip_xbasis:
            basis[:, 0] = -basis[:, 0]
        if flip_ybasis:
            basis[:, 1] = -basis[:, 1]

    basis = basis / np.linalg.norm(basis, axis=0)
    vertex_coordinates = vertex_ising @ basis

    class results:
        def __init__(self, vertex_coordinates, vertex_binaries, basis):
            self.vertex_coordinates = vertex_coordinates
            self.vertex_binaries = vertex_binaries
            self.basis = basis

    r = results(vertex_coordinates, vertex_binaries, basis)
    # if style == "pca":
    #    r.pca = pca
    # elif style == "wpca":
    if style == "wpca":
        r.pca = wpca
    else:
        r.pca = None
    return r


def _hypercube_edge_pairs(num_dimensions):
    """
    Generate hypercube edges. Returns a list of pairs of joined vertices. Each
    vertex is identified by its integral code.
    """
    edges = []
    for n in range(num_dimensions):
        vertices = 2**n
        new_edges = edges.copy()
        new_edges += [(i + vertices, j + vertices) for i, j in edges]
        new_edges += [(i, i + vertices) for i in range(vertices)]
        edges = new_edges
    return edges


def _hypercube_edge_pairs_from_vertices(vertices):
    distances = distance.pdist(vertices, "hamming")
    distances *= vertices.shape[1]
    square_distances = distance.squareform(distances)
    square_distances = np.triu(square_distances)
    edges = np.argwhere(square_distances == 1)
    return edges


def _hamiltonian_path_pairs(num_dimensions):
    binaries = _int2bin(range(2**num_dimensions), width=num_dimensions)
    grays = _bin2gray(binaries)
    int_from_gray = _bin2int(grays)
    changed = np.bitwise_xor(grays[:-1], grays[1:])
    changed_digit = np.argwhere(changed)[:, 1]
    colors = plt.cm.tab10(changed_digit)
    edges = [
        (i, j, c) for i, j, c in zip(int_from_gray[:-1], int_from_gray[1:], colors)
    ]
    return edges


def _draw_graph(
    coordinates,
    vertex_colors,
    vertex_cmap,
    plot_cbar,
    colorbar_location,
    pairs,
    labels,
    style,
    xcomponent,
    ycomponent,
    hamiltonian_path_pairs,
    basis_width,
    s,
    marker_lw,
    lw,
    plot_label,
    fontsize,
    label_margin,
    fig,
    ax,
    figsize,
    dpi,
):
    centroid = coordinates.mean(0) + 1e-8

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # hypercube vertices
    cbar = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=s,
        c=vertex_colors,
        cmap=vertex_cmap,
        lw=marker_lw,
        edgecolors="k",
        zorder=3,
    )
    if not isinstance(vertex_colors, str) and plot_cbar:
        cb = fig.colorbar(
            cbar,
            label="Energy",
            location=colorbar_location,
        )
        cb.outline.set_visible(False)

    # hypercube edges
    for i, j in pairs:
        ax.plot(
            [coordinates[i, 0], coordinates[j, 0]],
            [coordinates[i, 1], coordinates[j, 1]],
            lw=lw,
            color="k",
            zorder=0,
        )

    # Hamiltonian path
    if hamiltonian_path_pairs is not None:
        for i, j, c in hamiltonian_path_pairs:
            ax.arrow(
                coordinates[i, 0],
                coordinates[i, 1],
                coordinates[j, 0] - coordinates[i, 0],
                coordinates[j, 1] - coordinates[i, 1],
                fc=c,
                width=basis_width,
                length_includes_head=True,
                zorder=2,
                lw=0,
            )

    # labeling
    if plot_label:
        for point, label in zip(coordinates, labels):
            outer = point - centroid
            outer = outer / np.linalg.norm(outer)
            outer_angle = np.arctan2(outer[1], outer[0])
            ax.text(
                point[0] + outer[0] * label_margin + np.cos(outer_angle) * label_margin,
                point[1] + outer[1] * label_margin,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                color="k",
                zorder=3,
            )

    ax.set_aspect("equal")
    if not (style == "pca" or style == "wpca"):
        ax.axis("off")
    else:
        ax.set_xlabel("PC" + str(xcomponent + 1))
        ax.set_ylabel("PC" + str(ycomponent + 1))
    return fig, ax


def draw_basis(
    origin,
    basis,
    ax,
    basis_width,
    basis_cmap,
    color_by_angle,
    plot_basis_label,
    basis_label_fontsize=8,
    draw_by_patch=False,
):
    if color_by_angle:
        angles = np.arctan2(basis[:, 1], basis[:, 0])
        norm_angles = (angles + np.pi) / (2 * np.pi)
        colors = basis_cmap(norm_angles)
        angles = np.rad2deg(angles)
    else:
        colors = basis_cmap(np.arange(len(basis)))

    if draw_by_patch:
        # for xlim and ylim to be set correctly
        ax.scatter(
            origin[0] + basis[:, 0],
            origin[1] + basis[:, 1],
            s=0,
            alpha=0,
            lw=0,
        )
        arrows = []
        for u, v in zip(basis[:, 0], basis[:, 1]):
            arrow = mpl.patches.Arrow(origin[0], origin[1], u, v, width=basis_width)
            arrows.append(arrow)
        arrows = mpl.collections.PatchCollection(arrows)
        arrows.set_facecolor(colors)
        arrows.set_edgecolor(None)
        ax.add_collection(arrows)
    else:
        for i in range(len(basis)):
            ax.arrow(
                origin[0],
                origin[1],
                basis[i, 0],
                basis[i, 1],
                length_includes_head=True,
                zorder=2,
                width=basis_width,
                fc=colors[i],
                lw=0,
            )

    if plot_basis_label:
        label = np.arange(len(basis)) + 1
        for i in range(len(basis)):
            ax.text(
                basis[i, 0] + origin[0],
                basis[i, 1] + origin[1],
                f"{label[i]}",
                rotation=angles[i] - 90,
                rotation_mode="anchor",
                va="bottom",
                ha="center",
                fontsize=basis_label_fontsize,
            )
    return ax


def project_hypercube(
    num_dimensions=4,
    style="isometric",
    distortion=0,
    fractal_repeat=2,
    fractal_length_ratio=10,
    pca_ground_state_distance=None,
    xcomponent=0,
    ycomponent=1,
    weights=None,
    symmetric_weights=True,
    weights_threshold=None,
    flip_xbasis=False,
    flip_ybasis=False,
    plot_hamiltonian_path=False,
    swapped_id=None,
    return_pca=False,
    s=40,
    basis_cmap=plt.cm.tab10,
    vertex_cmap=plt.cm.viridis,
    vertex_colors=None,
    colorbar_location="right",
    plot_cbar=False,
    marker_lw=0,
    lw=0.4,
    plot_label=True,
    fontsize=6,
    label_margin=0.15,
    plot_basis=False,
    plot_basis_label=False,
    color_by_angle=False,
    basis_width=0.04,
    fig=None,
    ax=None,
    figsize=(3.4, 3.4),
    dpi=None,
):
    result = calc_hypercube_coordinates(
        num_dimensions,
        style,
        distortion,
        fractal_repeat,
        fractal_length_ratio,
        pca_ground_state_distance,
        xcomponent,
        ycomponent,
        weights,
        symmetric_weights,
        flip_xbasis,
        flip_ybasis,
    )
    vertex_coordinates = result.vertex_coordinates
    vertex_binaries = result.vertex_binaries
    basis = result.basis

    if style == "pca" or style == "wpca":
        pca = result.pca
    else:
        pca = None

    if plot_hamiltonian_path:
        hamiltonian_path_pairs = _hamiltonian_path_pairs(num_dimensions)
    else:
        hamiltonian_path_pairs = None

    if swapped_id is None:
        vertex_labels = _bin2labels(vertex_binaries)
    else:
        swapped_vertex_binaries = vertex_binaries.copy()
        for i in range(len(swapped_vertex_binaries)):
            swapped_vertex_binaries[i] = np.bitwise_xor(
                swapped_vertex_binaries[i], vertex_binaries[swapped_id[0]]
            )
            swapped_vertex_binaries[i] = np.bitwise_xor(
                swapped_vertex_binaries[i], vertex_binaries[swapped_id[1]]
            )
        vertex_labels = _bin2labels(swapped_vertex_binaries)

    if weights_threshold is not None:
        indices = weights >= weights_threshold
        weights = weights[indices]
        vertex_coordinates = vertex_coordinates[indices]
        vertex_binaries = vertex_binaries[indices]
        pairs = _hypercube_edge_pairs_from_vertices(vertex_binaries)
        vertex_labels = vertex_labels[indices]
        plot_basis = False
        if vertex_colors is None:
            potential = -np.log(weights)
            potential -= potential.min()
            potential /= potential.max()
            vertex_colors = vertex_cmap(potential)
        vertex_colors = vertex_colors[indices]
    else:
        pairs = _hypercube_edge_pairs(num_dimensions)
        vertex_colors = "r"

    fig, ax = _draw_graph(
        vertex_coordinates,
        vertex_colors,
        vertex_cmap,
        plot_cbar,
        colorbar_location,
        pairs,
        vertex_labels,
        style,
        xcomponent,
        ycomponent,
        hamiltonian_path_pairs,
        basis_width,
        s,
        marker_lw,
        lw,
        plot_label,
        fontsize,
        label_margin,
        fig,
        ax,
        figsize,
        dpi,
    )
    if plot_basis:
        origin = vertex_coordinates[0]
        if swapped_id is None:
            # multiply by 2 to adjust for Ising varialbes
            ax = draw_basis(
                origin,
                2 * basis,
                ax,
                basis_width,
                basis_cmap,
                color_by_angle,
                plot_basis_label,
            )
        else:
            changed_digit = np.bitwise_xor(
                vertex_binaries[swapped_id[0]], vertex_binaries[swapped_id[1]]
            )
            change = 2 * changed_digit - 1
            swapped_basis = basis * (-np.transpose((change, change)))
            # multiply by 2 to adjust for Ising varialbes
            origin += changed_digit @ basis * 2
            ax = draw_basis(
                origin,
                2 * swapped_basis,
                ax,
                basis_width,
                basis_cmap,
                color_by_angle,
                plot_basis_label,
            )
    if return_pca:
        return fig, ax, pca
    else:
        return fig, ax


def calc_dot_product_between_vertices(
    num_dimensions=4,
    style="isometric",
    distortion=0,
    fractal_repeat=2,
    fractal_length_ratio=10,
    pca_ground_state_distance=None,
    xcomponent=0,
    ycomponent=1,
    weights=None,
    symmetric_weights=True,
    flip_basis=False,
):
    # projection
    results = calc_hypercube_coordinates(
        num_dimensions,
        style,
        distortion,
        fractal_repeat,
        fractal_length_ratio,
        pca_ground_state_distance,
        xcomponent,
        ycomponent,
        weights,
        symmetric_weights,
        flip_basis,
    )
    coordinates = results.vertex_coordinates
    binaries = results.vertex_binaries
    isings = 2 * binaries - 1
    pca = results.pca

    # centering
    coordinates = coordinates - np.mean(np.vstack(weights) * coordinates, axis=0)
    isings = np.float64(isings)
    isings = isings - np.mean(np.vstack(weights) * isings, axis=0)

    # triu_indices = np.triu_indices(len(coordinates))

    dot_prod_projected = coordinates @ coordinates.T
    # dot_prod_projected = dot_prod_projected[triu_indices]

    dot_prod_isings = isings @ isings.T
    # dot_prod_isings = dot_prod_isings[triu_indices]

    if weights is None:
        weights_prod = np.ones((len(coordinates), len(coordinates)))
        weights_prod /= weights_prod.sum()
        # weights_prod = weights_prod[triu_indices]
    else:
        weights /= weights.sum()
        weights_prod = np.outer(weights, weights)
        # weights_prod = weights_prod[triu_indices]

    class results:
        def __init__(
            self,
            dot_prod_projected,
            dot_prod_isings,
            weights_prod,
            pca,
        ):
            self.dot_prod_projected = dot_prod_projected
            self.dot_prod_isings = dot_prod_isings
            self.weights_prod = weights_prod
            self.pca = pca

    r = results(dot_prod_projected, dot_prod_isings, weights_prod, pca)
    return r


def plot_flux_on_pca(
    weights,
    flux_edge_list,
    flux_scale=1,
    flux_threshold=None,
    xcomponent=0,
    ycomponent=1,
    basis_cmap=cc.cm.rainbow4,
    basis_origin=None,
    basis_width=0.2,
    color_by_angle=True,
    plot_basis_label=False,
    flip_xbasis=False,
    flip_ybasis=False,
    return_pca=False,
    fig=None,
    ax=None,
):
    num_dimensions = np.log2(len(weights)).astype(int)
    style = "wpca"
    result = calc_hypercube_coordinates(
        num_dimensions,
        style,
        xcomponent=xcomponent,
        ycomponent=ycomponent,
        weights=weights,
        flip_xbasis=flip_xbasis,
        flip_ybasis=flip_ybasis,
    )
    vertex_coordinates = result.vertex_coordinates
    basis = result.basis
    pca = result.pca

    if fig is None and ax is None:
        fig, ax = plt.subplots()
    # for xlim and ylim to be set correctly
    ax.scatter(
        vertex_coordinates[:, 0],
        vertex_coordinates[:, 1],
        s=0,
        alpha=0,
        lw=0,
    )

    flux_source = flux_edge_list[0]
    flux_target = flux_edge_list[1]
    flux_rates = flux_edge_list[2]
    flux_width = flux_rates * flux_scale
    change = np.argwhere(flux_source * flux_target == -1)[:, 1]
    if color_by_angle:
        angles = np.arctan2(basis[:, 1], basis[:, 0])
        norm_angles = (angles + np.pi) / (2 * np.pi)
        colors = basis_cmap(norm_angles)
        vec_color = colors[change]
    else:
        colors = plt.cm.tab10
        vec_color = colors(change)
    source_pos = flux_source @ basis
    target_pos = flux_target @ basis
    arrow_vec = target_pos - source_pos

    X = source_pos[:, 0]
    Y = source_pos[:, 1]
    U = arrow_vec[:, 0]
    V = arrow_vec[:, 1]

    # Sort flux by strength
    flux_sorted_indices = np.argsort(flux_width)[::-1]
    X = X[flux_sorted_indices]
    Y = Y[flux_sorted_indices]
    U = U[flux_sorted_indices]
    V = V[flux_sorted_indices]
    flux_width = flux_width[flux_sorted_indices]
    vec_color = vec_color[flux_sorted_indices]

    if flux_threshold is not None:
        indices = flux_width >= flux_threshold
        X = X[indices]
        Y = Y[indices]
        U = U[indices]
        V = V[indices]
        flux_width = flux_width[indices]
        vec_color = vec_color[indices]

    arrows = []
    for x, y, u, v, w in zip(X, Y, U, V, flux_width):
        arrow = mpl.patches.Arrow(x, y, u, v, width=w * basis_width)
        arrows.append(arrow)
    arrows = mpl.collections.PatchCollection(arrows)
    arrows.set_facecolor(vec_color)
    arrows.set_edgecolor(None)
    ax.add_collection(arrows)

    # multiply by 2 to adjust for Ising varialbes
    if basis_origin is None:
        basis_origin = np.array([2.5, 2.5])
    if not color_by_angle:
        basis_cmap = plt.cm.tab10
    ax = draw_basis(
        basis_origin,
        2 * basis,
        ax,
        basis_width,
        basis_cmap,
        color_by_angle=color_by_angle,
        plot_basis_label=plot_basis_label,
        draw_by_patch=True,
    )

    ax.set_xlabel("PC{}".format(xcomponent + 1))
    ax.set_ylabel("PC{}".format(ycomponent + 1))
    ax.set_aspect("equal")
    if return_pca:
        return fig, ax, pca
    else:
        return fig, ax
