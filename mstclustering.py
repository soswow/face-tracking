from __future__ import division

import networkx as nx
import itertools
import math

from cvutils import *
from swipeline import calculate_area


def draw_graph(img, G, points, color=cv.RGB(0,255,255),thickness=1):
    for start, end in G.edges():
        start_pt = (points[start][0], points[start][1])
        end_pt = (points[end][0], points[end][1])
        cv.Line(img, start_pt, end_pt,color,thickness)

def remove_longest_edge(G):
    edges = G.edges(data=True)
    edges.sort(key=lambda a:a[2]['weight'],reverse=True)
    G.remove_edge(edges[0][0],edges[0][1])

def hemst(G, k):
    nc = 1
    mst = G
    point_set = {}
    while nc != k:
        nc = 1
        mst = nx.minimum_spanning_tree(mst)
        weights = np.array([attrs['weight'] for _,_,attrs in mst.edges(data=True)])
        mean_w = weights.mean()
        std = weights.std()

        for a,b,attrs in mst.edges(data=True):
            w = attrs['weight']
            if w > mean_w + std:
                mst.remove_edge(a,b)
                nc+=1

        if nc < k:
            while nc != k:
                remove_longest_edge(mst)
                nc+=1
            break

        if nc > k:
            sG = nx.connected_component_subgraphs(mst)
            centroid_nodes = []
            for g in sG:
                cl = nx.closeness_centrality(g)
                sorted_set_nodes = sorted(cl.items(), key=lambda a: a[1])
                closest_to_c = sorted_set_nodes[0][0]

                point_set[closest_to_c] = g.nodes()
                for p, _ in sorted_set_nodes[1:]:
                    if p in point_set:
                        point_set[closest_to_c]+= point_set[p]

                centroid_nodes.append(closest_to_c)

            edges=itertools.combinations(centroid_nodes,2)
            mst.clear()
            mst.add_nodes_from(centroid_nodes)
            mst.add_edges_from(edges)
            for u,v in mst.edges():
                weight = G.get_edge_data(u,v)["weight"]
                nx.set_edge_attributes(mst, "weight", {(u,v):weight})
                
    sG = nx.connected_component_subgraphs(mst)
    if point_set:
        for g in sG:
            for node in g.nodes():
                if node in point_set:
                    g.add_nodes_from(point_set[node])

    return sG

def get_hemst_clusters(verticies, k=3):
    G = nx.complete_graph(len(verticies))
    for edge in G.edges():
        dist = np.linalg.norm(verticies[edge[0]]-verticies[edge[1]])
        nx.set_edge_attributes(G, "weight", {edge:dist})

    mst = nx.minimum_spanning_tree(G)

    cluster_forest = hemst(G, k)

    node_cluster_mapping = {}
    cluster_nodes_mapping = {}
    for i, tree in enumerate(cluster_forest):
        for node in tree.nodes():
            node_cluster_mapping[node] = i
            if i not in cluster_nodes_mapping:
                cluster_nodes_mapping[i] = []
            cluster_nodes_mapping[i].append(node)

    return node_cluster_mapping, cluster_nodes_mapping, mst

def get_color(i):
    colors = [cv.RGB(255,10,10),
             cv.RGB(255,255,10),
             cv.RGB(10,255,255),
             cv.RGB(255,10,255),
             cv.RGB(255,100,100),
             cv.RGB(255,255,100),
             cv.RGB(100,255,255),
             cv.RGB(255,100,255)]
    if i >= len(colors):
        return get_color(i-len(colors))
    return colors[i]

def get_corners_map(boxes):
    corners = {}
    for i, (x,y,w,h) in enumerate(boxes):
        corners[i] = [(x,y), (x+w,y), (x,y+h), (x+w,y+h)]
    return corners

def merge_boxes(boxes,img=None):
    if img:
        orig = cv.CloneImage(img)
    threshold = 0.7
    loan = 0.0
    init_size = 0
    while True:
        for k in range(len(boxes)-1, 0, -1):
            if threshold > 1:
                threshold = 0.95
            corners = get_corners_map(boxes)
            if img:
                img = cv.CloneImage(orig)
                draw_boxes(boxes, img)
            verticies = np.array([(x+w/2, y+h/2) for x,y,w,h in boxes])
            if len(verticies) < k:
                continue #Not enough verticies for K clusters
            node_cluster_map, cluster_nodes_map, mst = get_hemst_clusters(verticies, k)

            for_merge = []
            for i, nodes in cluster_nodes_map.items():
                cluster_points = []
                if len(nodes) > 1:
                    for id in nodes:
                        cluster_points += corners[id]
                    current_boxes = [boxes[id] for id in nodes]

                    new_box = x,y,w,h = cv.BoundingRect(cluster_points)

                    new_rect_area = w*h
                    old_rects_area = calculate_area(current_boxes)

                    relation = math.sqrt(old_rects_area)/math.sqrt(new_rect_area)
                    if relation > threshold:
                        loan += 1-relation
                        for_merge.append((current_boxes, new_box))

                    if img:
                        print "Threshold = %s" % threshold
                        print "k=%d, was area=%d, become=%d, %%%.3f (%d rects)" % \
                              (k, old_rects_area, new_rect_area, relation, len(current_boxes))
                        cv.Rectangle(img, (x,y), (x+w,y+h), color=get_color(k))

            if for_merge:
#                print "Loan = %.2f" % loan
                threshold += 0.035 + loan/k
            for merge_boxes, new_box in for_merge:
                boxes = [box for box in boxes if box not in merge_boxes]
                boxes.append(new_box)

            #For drawing only
            if img:
                draw_graph(img, mst, verticies, color=cv.RGB(255,0,255), thickness=2)
                show_image(img)

        if len(boxes) == init_size:
            break
        init_size = len(boxes)
    return boxes

def main():
    pass

#    for i,xy in enumerate(verticies):
#        cv.Circle(img, tuple(xy), 5, colors[clusters[i]], thickness=-1)



if __name__ == "__main__":
    main()
  