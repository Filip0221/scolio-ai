import numpy as np

# Liczy kąt Cobba między dwiema liniami.
def cobb_angle(line1, line2):
    (x1, y1), (x2, y2), _ = line1
    (x3, y3), (x4, y4), _ = line2
    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x4 - x3, y4 - y3])
    v1p = np.array([-v1[1], v1[0]])
    v2p = np.array([-v2[1], v2[0]])
    ang = np.degrees(np.arccos(np.clip(np.dot(v1p, v2p) / (np.linalg.norm(v1p)*np.linalg.norm(v2p)), -1, 1)))
    return ang if ang <= 90 else 180 - ang

# Tworzy prostą prostopadłą do odcinka p1–p2, przechodzącą przez 'through'.
def line_perpendicular(p1, p2, through):
    x1, y1 = p1; x2, y2 = p2; xt, yt = through
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    perp = np.array([-dy, dx]) / np.linalg.norm([dx, dy])
    L = 2000
    return (xt - perp[0]*L, yt - perp[1]*L), (xt + perp[0]*L, yt + perp[1]*L)

# Liczy punkt przecięcia dwóch kierunków P+tu i Q+sv.
def intersection_of_dirs(P, u, Q, v):
    A = np.array([[u[0], -v[0]], [u[1], -v[1]]], float)
    b = np.array([Q[0]-P[0], Q[1]-P[1]], float)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t, _ = np.linalg.solve(A, b)
    return (P[0] + t*u[0], P[1] + t*u[1])
