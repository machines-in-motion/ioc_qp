from numpy import hstack, vstack
from quadprog import solve_qp

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
    using quadprog <https://pypi.python.org/pypi/quadprog/>.
    Parameters
    ----------
    P : numpy.array
        Symmetric quadratic-cost matrix.
    q : numpy.array
        Quadratic-cost vector.
    G : numpy.array
        Linear inequality constraint matrix.
    h : numpy.array
        Linear inequality constraint vector.
    A : numpy.array, optional
        Linear equality constraint matrix.
    b : numpy.array, optional
        Linear equality constraint vector.
    initvals : numpy.array, optional
        Warm-start guess vector (not used).
    Returns
    -------
    x : numpy.array
        Solution to the QP, if found, otherwise ``None``.
    Note
    ----
    The quadprog solver only considers the lower entries of `P`, therefore it
    will use a wrong cost function if a non-symmetric matrix is provided.
    """
    if initvals is not None:
        print("quadprog: note that warm-start values ignored by wrapper")
    qp_G = P
    qp_a = -q
    if A is not None:
        qp_C = -vstack([A, G]).T
        qp_b = -hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
        
    x_opt, _, _, _, lag, iact = solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
    return x_opt, lag[A.shape[0]:]