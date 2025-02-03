import numpy as np
def is_in_hull(point, hull, eps=None):
    """Check if point is in convex hull within epsilon tolerance.

    >>> import torch
    >>> from scipy.spatial import ConvexHull
    >>> pts = torch.rand(100, 4)
    >>> hull = ConvexHull(pts)
    >>> all(is_in_hull(pts[i,:], hull) for i in range(pts.shape[0]))
    True
    """
    point = np.asarray(point)
    eps = (eps if eps else np.sqrt(np.finfo(point.dtype).eps))
    equations = hull.equations  # (normal_vector, offset) for each facet
    return np.all(np.dot(equations[:, :-1], point) + equations[:, -1] <=eps)
