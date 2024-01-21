import math
import numpy as np


def neighbors_dist(neighbors, k1, k2):
    a = k1[0] - k2[0]
    b = k1[1] - k2[1]
    return math.sqrt(a*a+b*b)


def neighbors_norm(neighbors, k1, k2):
    l = neighbors_dist(neighbors, k1, k2)
    a = k1[0] - k2[0]
    b = k1[1] - k2[1]
    return a/l, b/l


def neighbors_cos(neighbors, k1, k2, k3):
    vec1 = neighbors_norm(neighbors, k2, k1)
    vec2 = neighbors_norm(neighbors, k3, k1)
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]


def ccw(a, b, c):
    return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])


def intersect(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def distance(k1, k2):
    a = k1[0]-k2[0]
    b = k1[1]-k2[1]
    return np.sqrt(a*a + b*b)


def intersect_point(A, B, C, D):
    l = distance(A, B)

    min_d = 100000000000
    min_p = A

    for i in range(int(l)):
        a = float(i)/l
        x = A[0] * (1-a) + B[0]*a
        y = A[1] * (1-a) + B[1]*a

        d = distance((x, y), C)
        d += distance((x, y), D)

        if d < min_d:
            min_d = d
            min_p = (x, y)

    return min_p


def graph_coverage(nei, p, r=4):
    visited = {}
    depth = {}
    queue = [p]
    depth[p] = 0
    visited[p] = 1

    while len(queue) > 0:
        cp = queue.pop()

        if depth[cp] > r:
            continue

        for n in nei[cp]:
            if n not in visited:
                depth[n] = depth[cp] + 1
                queue.append(n)
                visited[n] = 1

    return visited.keys()
