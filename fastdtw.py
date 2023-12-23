import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import sparse
import time
from numpy import linalg
import warnings
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def check_euclidean_inputs(X, Y):
    """
    Check the input of two time series in Euclidean spaces, which
    are to be warped to each other.  They must satisfy:
    1. They are in the same dimension space
    2. They are 32-bit
    3. They are in C-contiguous order

    If #2 or #3 are not satisfied, automatically fix them and
    warn the user.
    Furthermore, warn the user if X or Y has more columns than rows,
    since the convention is that points are along rows and dimensions
    are along columns

    Parameters
    ----------
    X: ndarray(M, d)
        The first time series
    Y: ndarray(N, d)
        The second time series

    Returns
    -------
    X: ndarray(M, d)
        The first time series, possibly copied in memory to be 32-bit, C-contiguous
    Y: ndarray(N, d)
        The second time series, possibly copied in memory to be 32-bit, C-contiguous
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("The input time series are not in the same dimension space")
    if X.shape[0] < X.shape[1]:
        warnings.warn("X {} has more columns than rows; did you mean to transpose?".format(X.shape))
    if Y.shape[0] < Y.shape[1]:
        warnings.warn("Y {} has more columns than rows; did you mean to transpose?".format(Y.shape))
    if not X.dtype == np.float32:
        warnings.warn("X is not 32-bit, so creating 32-bit version")
        X = np.array(X, dtype=np.float32)
    if not X.flags['C_CONTIGUOUS']:
        warnings.warn("X is not C-contiguous; creating a copy that is C-contiguous")
        X = X.copy(order='C')
    if not Y.dtype == np.float32:
        warnings.warn("Y is not 32-bit, so creating 32-bit version")
        Y = np.array(Y, dtype=np.float32)
    if not Y.flags['C_CONTIGUOUS']:
        warnings.warn("Y is not C-contiguous; creating a copy that is C-contiguous")
        Y = Y.copy(order='C')
    return X, Y


def fill_block(A, p, radius, val):
    """
    Fill a square block with values

    Parameters
    ----------
    A: ndarray(M, N) or sparse(M, N)
        The array to fill
    p: list of [i, j]
        The coordinates of the center of the box
    radius: int
        Half the width of the box
    val: float
        Value to fill in
    """

    move_path = []
    for i in range(p.shape[0] - 1):
        loc = p[i]
        next_loc = p[i + 1]
        if loc[0] == next_loc[0]:
            # move along row
            move_path.extend(
                [[2 * loc[0], 2 * loc[1]], [2 * loc[0], 2 * loc[1] + 1], [2 * next_loc[0], 2 * next_loc[1]]])
        elif loc[1] == next_loc[1]:
            # move along column
            move_path.extend(
                [[2 * loc[0], 2 * loc[1]], [2 * loc[0] + 1, 2 * loc[1]], [2 * next_loc[0], 2 * next_loc[1]]])
        else:
            move_path.extend(
                [[2 * loc[0], 2 * loc[1]], [2 * loc[0] + 1, 2 * loc[1] + 1], [2 * next_loc[0], 2 * next_loc[1]]])
    # projecting
    for path_start in move_path:
        A[path_start[0]:path_start[0] + 2, path_start[1]:path_start[1] + 2] = val
    # expanding radius
    in_radius_index = []
    # print(A.toarray())
    for r in range(1, radius + 1):
        bottom_left = np.where((np.transpose(A.nonzero()) + [r, -r]) < 0, 0, (np.transpose(A.nonzero()) + [r, -r]))
        # print("Bottom left, first",bottom_left)
        bottom_left = np.where(bottom_left < A.shape[0], bottom_left, A.shape[0] - 1)
        # print("Bottom left, second",bottom_left)
        in_radius_index.extend(bottom_left.tolist())
        top_right = (np.where((np.transpose(A.nonzero()) + [-r, r]) < 0, 0, (np.transpose(A.nonzero()) + [-r, r])))
        top_right = np.where(top_right < A.shape[1], top_right, A.shape[1] - 1)
        in_radius_index.extend(top_right.tolist())
    for index in in_radius_index:
        A[index[0], index[1]] = max(0.5, A[index[0], index[1]])


def dtw(X, Y):
    X, Y = check_euclidean_inputs(X, Y)
    
    M = X.shape[0]
    N = Y.shape[0]
    d = Y.shape[1]
    P = np.zeros((M, N), dtype='int32')


    U = np.zeros((1, 1), dtype='float32')
    L = np.zeros((1, 1), dtype='float32')
    UL = np.zeros((1, 1), dtype='float32')
    S = np.zeros((M, N), dtype='float32')


    for i in range(M):
        for j in range(N):
            # step 1: compute the euclidean
            dist = calculate_distance(X[i], Y[j],)
             
            # step2: do dynamic programming
            score = -1
            LEFT = 0
            UP = 1
            DIAG = 2
            if (i == 0) & (j == 0):
                score = 0
            else:
                # left
                left = -1
                if j > 0:
                    left = S[i, (j - 1)]
                # up
                up = -1
                if i > 0:
                    up = S[(i - 1), j]
                #diag
                diag = -1
                if i > 0 and j > 0:
                    diag = S[(i - 1), (j - 1)]
                #filling the matrix 
                if left > -1:
                    score = left
                    P[i, j] = LEFT
                if (up > -1) and (up < score or score == -1):
                    score = up
                    P[i, j] = UP
                if (diag > -1) and (diag <= score or score == -1):
                    score = diag
                    P[i, j] = DIAG
            S[i, j] = score + dist
    dist = S[-1, -1]
    ret = {'cost': dist, 'P': P}
    return ret


 

def dtw_brute_backtrace(X, Y , do_plot=False, usage="dtw"):
    """
    Compute dynamic time warping between two time-ordered
    point clouds in Euclidean space, using cython on the
    backend.  Then, trace back through the matrix of backpointers
    to extract an alignment path

    Parameters
    ----------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points

    Returns
    -------
     (float: cost, ndarray(K, 2): The warping path)


    """
    res = dtw(X, Y)
    res['P'] = np.asarray(res['P'])

    i = X.shape[0] - 1
    j = Y.shape[0] - 1
    path = [[i, j]]
    step = [[0, -1], [-1, 0], [-1, -1]]  # LEFT, UP, DIAG
    while not (path[-1][0] == 0 and path[-1][1] == 0):
        s = step[res['P'][i, j]]
        i += s[0]
        j += s[1]
        path.append([i, j])
    path.reverse()
    path = np.array(path, dtype=int)


    if do_plot:  # pragma: no cover
        fig, ax = plt.subplots(figsize=(8, 8) )


        from matplotlib import colors
        cmap = colors.ListedColormap(['white', '#454545'])
        Occ = np.ones([X.shape[0], Y.shape[0]])
        # Occ[path[:,0],path[:,1]] = 1

        ax.imshow(Occ, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)

        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1.5)
        ax.set_xticks(np.arange(-.5, Occ.shape[0], 1));
        ax.set_yticks(np.arange(-.5, Occ.shape[1], 1));
        ax.set_xticklabels([]);
        ax.set_yticklabels([]);
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], c='black', linewidth=2, label="warp path")
        line_legend = mlines.Line2D([], [], color="black", label="warp path")

        if usage == "dtw":
            ax.set_xticklabels([]);
            ax.set_yticklabels([]);
            for i in range(Occ.shape[0]):
                ax.text(-1, i, str(i), ha='center', va='center', color='black')
            for j in range(Occ.shape[1]):
                ax.text(j, Occ.shape[0], str(j), ha='center', va='center', color='black')
            black_patch = mpatches.Patch(facecolor="#454545", label="search area", edgecolor="black")
            ax.legend(handles=[black_patch, line_legend], bbox_to_anchor=(1.05, 1),
                      loc="upper left")
            plt.title("Dynamic Warping Paths")
            plt.savefig("DTW plot.png", bbox_inches='tight')
        else:
            grey_patch = mpatches.Patch(facecolor="grey", label="search window-by radius", edgecolor="black")
            black_patch = mpatches.Patch(facecolor="#454545", label="search window-by projection", edgecolor="black")
            white_patch = mpatches.Patch(facecolor="white", label="unsearched area", edgecolor="black")
            ax.legend(handles=[grey_patch, black_patch, white_patch, line_legend], bbox_to_anchor=(1.05, 1),
                      loc="upper left")
            plt.title("Last Level")
            plt.savefig("Last Level.pdf", bbox_inches='tight')
    return (res['cost'], path)


def calculate_distance(series_a, series_b, method='l-2'):
    # use lp distance, with 1
    p = method.split("-")[-1]
    if p.isdigit():
        distance = linalg.norm(series_a - series_b, ord=int(p))
    elif p == 'inf':
        distance = (linalg.norm(series_a - series_b, ord=np.inf))
    return distance


def fastdtw_dynstep(X, Y, I, J, left, up, diag, S, P, distance_method='l-2'):
    """
    :param X:
    :param Y:
    :param d:
    :param I:
    :param J:
    :param left:
    :param up:
    :param diag:
    :param S:
    :param P:
    :param N:
    :return:
    """
    N = left.size
    for idx in range(N):

        # Step 1: Compute Euclidean distance
        ##Some code here, to calculate the euclidean distance named dist.
        # calculate_distance is a self defined distance metric, that take two data point or two vector with equal length and return a scaler.
        # dist = calculate_distance(X[idx//(Y.shape[0])],Y[idx%(Y.shape[0])],distance_method)
        dist = calculate_distance(X[I[idx]], Y[J[idx]], distance_method)
        # Step 2: Do dynamic programming step
        score = -1  # initialize to -1
        LEFT = 0
        UP = 1
        DIAG = 2
        if idx == 0:
            score = 0
        else:
            left_score = -1
            if left[idx] >= 0:
                left_score = S[left[idx]]
            up_score = -1
            if up[idx] >= 0:
                up_score = S[up[idx]]
            diag_score = -1
            if diag[idx] >= 0:
                diag_score = S[diag[idx]]

            if left_score > -1:
                score = left_score
                P[idx] = LEFT

            if (up_score > -1) and (up_score <= score or score == -1):
                score = up_score
                P[idx] = UP
            if (diag_score > -1) and (diag_score <= score or score == -1):
                score = diag_score
                P[idx] = DIAG
        # breakpoint()
        S[idx] = score + dist
        # print(S)


def get_path_cost(X, Y, path):
    """
    Return the cost of a warping path that matches two Euclidean
    point clouds

    Parameters
    ---------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    P1: ndarray(K, 2)
        Warping path

    Returns
    -------
    cost: float
        The sum of the Euclidean distances along the warping path
        between X and Y
    """
    x = X[path[:, 0], :]
    y = Y[path[:, 1], :]
    ds = np.sqrt(np.sum((x - y) ** 2, 1))
    return np.sum(ds)


def _dtw_constrained_occ(X, Y, Occ , level=0, do_plot=False):
    """
    DTW on a constrained occupancy mask.  A helper method for  fastdtw


    Parameters
    ----------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    Occ: scipy.sparse((M, N))
        A MxN array with 1s if this cell is to be evaluated and 0s otherwise

    level: int
        An int for keeping track of the level of recursion, if applicable
    do_plot: boolean
        Whether to plot the warping path at each level and save to image files

    Returns
    -------
        (float: cost, ndarray(K, 2): The warping path)
    """
    M = X.shape[0]
    N = Y.shape[0]
    tic = time.time()
    S = sparse.lil_matrix((M, N))
    P = sparse.lil_matrix((M, N), dtype=int)
    I, J = Occ.nonzero()
    # Sort cells in raster order
    idx = np.argsort(J)
    I = I[idx]
    J = J[idx]
    idx = np.argsort(I, kind='stable')
    I = I[idx]
    J = J[idx]

    ## Step 2: Find indices of left, up, and diag neighbors.
    # All neighbors must be within bounds *and* within sparse structure
    # Make idx M+1 x N+1 so -1 will wrap around to 0
    # Make 1-indexed so all valid entries have indices > 0
    idx = sparse.coo_matrix((np.arange(I.size) + 1, (I, J)), shape=(M + 1, N + 1)).tocsr()
    # idx is all the indices of all non-zero elements in the matrix.

    # Left neighbors
    left = np.array(idx[I, J - 1], dtype=np.int32).flatten()
    left[left <= 0] = -1
    left -= 1
    # Up neighbors
    up = np.array(idx[I - 1, J], dtype=np.int32).flatten()
    up[up <= 0] = -1
    up -= 1
    # Diag neighbors
    diag = np.array(idx[I - 1, J - 1], dtype=np.int32).flatten()
    diag[diag <= 0] = -1
    diag -= 1

    ## Step 3: Pass information for dynamic programming steps
    S = np.zeros(I.size, dtype=np.float32)  # Dyn prog matrix
    P = np.zeros(I.size, dtype=np.int32)  # Path pointer matrix
    fastdtw_dynstep(X, Y, I, J, left, up, diag, S, P)
    P = sparse.coo_matrix((P, (I, J)), shape=(M, N)).tocsr()
    if  do_plot:  # pragma: no cover
        S = sparse.coo_matrix((S, (I, J)), shape=(M, N)).tocsr()

    # Step 4: Do backtracing
    i = M - 1
    j = N - 1

    path = [[i, j]]
    step = [[0, -1], [-1, 0], [-1, -1]]  # LEFT, UP, DIAG
    while not (path[-1][0] == 0 and path[-1][1] == 0):
        s = step[P[i, j]]
        i += s[0]
        j += s[1]
        path.append([i, j])
    path.reverse()
    path = np.array(path, dtype=int)

    if do_plot:  # pragma: no cover
        fig, ax = plt.subplots(figsize=(8, 8) )


        cmap = colors.ListedColormap(['white', "grey", '#454545'])
        ax.imshow(Occ.toarray(), cmap=cmap, interpolation='nearest', vmin=0, vmax=1)

        grey_patch = mpatches.Patch(facecolor="grey", label="search window-by radius", edgecolor="black")
        black_patch = mpatches.Patch(facecolor="#454545", label="search window-by projection", edgecolor="black")
        white_patch = mpatches.Patch(facecolor="white", label="unsearched area", edgecolor="black")
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1.5)
        ax.set_xticks(np.arange(-.5, Occ.shape[1], 1));
        ax.set_yticks(np.arange(-.5, Occ.shape[0], 1));
        ax.set_xticklabels([]);
        ax.set_yticklabels([]);
        # for i in range(Occ.shape[0]):
        #     ax.text(-1, i, str(i), ha='center', va='center', color='black')
        # for j in range(Occ.shape[1]):
        #     ax.text(j, Occ.shape[0], str(j), ha='center', va='center', color='black')
        path = np.array(path)
        ax.plot(path[:, 1], path[:, 0], c='black', linewidth=2, label="warp path")
        line_legend = mlines.Line2D([], [], color="black", label="warp path")
        ax.legend(handles=[grey_patch, black_patch, white_patch, line_legend], bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.title("Level {}".format(level))
        plt.savefig("Level_%i.pdf" % level, bbox_inches='tight')


        ##Plot paths and series
        # fig  = plt.figure(figsize = (40,20))
        # ax1 = plt.subplot(1,2,1)
        # ax2 = plt.subplot(1,2,2)
        # # plt.figure(figsize=(8, 8))
        #
        # cmap = colors.ListedColormap(['white', "grey", '#454545'])
        #
        # ax1.imshow(Occ.toarray(), cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
        #
        # grey_patch = mpatches.Patch(facecolor="grey", label="search window-by radius", edgecolor="black")
        # black_patch = mpatches.Patch(facecolor="#454545", label="search window-by projection", edgecolor="black")
        # white_patch = mpatches.Patch(facecolor="white", label="unsearched area", edgecolor="black")
        # ax1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1.5)
        # ax1.set_xticks(np.arange(-.5, Occ.shape[1], 1));
        # ax1.set_yticks(np.arange(-.5, Occ.shape[0], 1));
        # ax1.set_xticklabels([]);
        # ax1.set_yticklabels([]);
        # for i in range(Occ.shape[0]):
        #     ax1.text(-1, i, str(i), ha='center', va='center', color='black')
        # for j in range(Occ.shape[1]):
        #     ax1.text(j, Occ.shape[0], str(j), ha='center', va='center', color='black')
        # path = np.array(path)
        # ax1.plot(path[:, 1], path[:, 0], c='black', linewidth=2, label="warp path")
        # line_legend = mlines.Line2D([], [], color="black", label="warp path")
        # ax1.legend(handles=[grey_patch, black_patch, white_patch, line_legend], bbox_to_anchor=(1.05, 1),
        #           loc="upper left")
        # ax2.plot(X,c='r',label = 'X')
        # ax2.plot(Y,c = 'b',label = 'Y')
        # ax2.legend()
        #
        # plt.title("Level {}".format(level))
        # plt.savefig("Level_%i.png" % level, bbox_inches='tight')
    return (get_path_cost(X, Y, path), path)



def reduce_by_half(series):
    X = series.copy()
    if X.shape[0] % 2 != 0:
        X[-2] = (X[-2] + X[-1]) / 2
        X = X[:-1]
    return (X[1::2] + X[::2]) / 2


def fastdtw(X, Y, radius, level=0, do_plot=False):
    """
    An implementation of [1]
    [1] FastDTW: Toward Accurate Dynamic Time Warping in Linear Time and Space. Stan Salvador and Philip Chan

    Parameters
    ----------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    radius: int
        Radius of the l-infinity box that determines sparsity structure
        at each level
    level: int
        An int for keeping track of the level of recursion
    do_plot: boolean
        Whether to plot the warping path at each level and save to image files

    Returns
    -------
        (float: cost, ndarray(K, 2): The warping path)
    """
    X, Y = check_euclidean_inputs(X, Y)
    minTSsize = radius + 2
    M = X.shape[0]
    N = Y.shape[0]
    X = np.ascontiguousarray(X)
    Y = np.ascontiguousarray(Y)
    if M < minTSsize or N < minTSsize:
        return dtw_brute_backtrace(X, Y,do_plot= do_plot, usage="fastdtw")
    # Recursive step
    shrunk_x = reduce_by_half(X)
    shrunk_y = reduce_by_half(Y)
    cost, path = fastdtw(shrunk_x, shrunk_y, radius,  level + 1, do_plot)
    # cost, path = fastdtw(X[0::2, :], Y[0::2, :], radius,  level + 1, do_plot)
    if type(path) is dict:
        path = path['path']
    path = np.array(path)
    Occ = sparse.lil_matrix((M, N))
    fill_block(Occ, path, radius, 1)
    return _dtw_constrained_occ(X, Y, Occ,  level, do_plot)


if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(16, 1)
    Y = np.random.rand(16, 1)
    import time

    tic = time.time()
    cost_fast, path_fast = fastdtw(X, Y, radius=2, level=0, do_plot=True)
    toc = time.time()
    print(
        "Implementing FastDTW, on X with {} data points and Y with {} data points.\n-Time Use: {}s\n-Distance: {}".format(
            X.shape[0], Y.shape[0], tic - toc, cost_fast))
    tic = time.time()
    breakpoint()
    cost_dtw, path_dtw = dtw_brute_backtrace(X, Y, do_plot=True)
    toc = time.time()
    print(
        "Implementing OriginalDTW, on X with {} data points and Y with {} data points.\n-Time Use: {}s\n-Distance: {}".format(
            X.shape[0], Y.shape[0], tic - toc, cost_dtw))
    # print(path_dtw)
