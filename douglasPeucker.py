import math
import cv2
from PIL import Image

def distance(p1, p2):
    a = p1[0] - p2[0]
    b = p1[1] - p2[1]

    return math.sqrt(a * a + b * b)


def point2line_distance(p, n1, n2):
    l = distance(n1, n2) + 0.00000001

    v1 = [n1[0] - p[0], n1[1] - p[1]]
    v2 = [n2[0] - p[0], n2[1] - p[1]]

    area = abs(v1[0] * v2[1] - v1[1] * v2[0])

    return area / l


# simplifies the graph while still retaining the essential shape
def douglas_peucker(node_list, e=5):

    if len(node_list) <= 2:
        return node_list

    best_i = 1
    best_d = 0

    for i in range(1, len(node_list) - 1):
        d = point2line_distance(node_list[i], node_list[0], node_list[-1])
        if d > best_d:
            best_d = d
            best_i = i

    if best_d <= e:
        return [node_list[0], node_list[-1]]

    new_list = douglas_peucker(node_list[0:best_i + 1], e=e)
    new_list = new_list[:-1] + douglas_peucker(node_list[best_i:len(node_list)], e=e)

    return new_list


# establishes edge connection between the two nodes: n1key and n2key
def graph_insert(node_neighbor, n1key, n2key):
    if n1key != n2key:
        if n1key in node_neighbor:
            if n2key in node_neighbor[n1key]:
                pass
            else:
                node_neighbor[n1key].append(n2key)
        else:
            node_neighbor[n1key] = [n2key]

        if n2key in node_neighbor:
            if n1key in node_neighbor[n2key]:
                pass
            else:
                node_neighbor[n2key].append(n1key)
        else:
            node_neighbor[n2key] = [n1key]

    return node_neighbor


def simplify_graph(node_neighbor, e=2.5):
    visited = []

    new_node_neighbor = {}

    for node, node_nei in node_neighbor.items():
        if len(node_nei) == 1 or len(node_nei) > 2:
            if node in visited:
                continue

            for next_node in node_nei:
                if next_node in visited:
                    continue

                node_list = [node, next_node]

                while True:
                    if len(node_neighbor[node_list[-1]]) == 2:
                        if node_neighbor[node_list[-1]][0] == node_list[-2]:
                            node_list.append(node_neighbor[node_list[-1]][1])
                        else:
                            node_list.append(node_neighbor[node_list[-1]][0])
                    else:
                        break

                for i in range(len(node_list) - 1):
                    if node_list[i] not in visited:
                        visited.append(node_list[i])

                # simplify node_list
                new_node_list = douglas_peucker(node_list, e=e)

                for i in range(len(new_node_list) - 1):
                    new_node_neighbor = graph_insert(new_node_neighbor, new_node_list[i], new_node_list[i + 1])

    return new_node_neighbor


def draw_graph(graph, input_sat):
    simplified_graph = simplify_graph(graph)
    for k, v in simplified_graph.items():
        n1 = k
        for n2 in v:
            cv2.line(input_sat, (n1[1], n1[0]), (n2[1], n2[0]), (255, 255, 0), 3)

    return input_sat
