import cv
import itertools
import numpy as np
import networkx as nx
import itertools

from cvutils import *
from contours import *


def get_image_w_boxes():
#    img = cv.LoadImage("/Users/soswow/Documents/Face Detection/Face Detection Data Set and Benchmark"
#                       "/originalPics/2002/07/19/big/cv/bad/img_554.jpg")
#    img = cv.LoadImage("/Users/soswow/Documents/Face Detection/Face Detection Data Set and Benchmark"
#                       "/originalPics/2002/07/22/big/img_500.jpg")
#                       "/originalPics/2002/07/22/big/img_570.jpg")
#                       "/originalPics/2002/07/22/big/img_696.jpg")
#                       "/originalPics/2002/07/22/big/img_835.jpg")
    img = cv.LoadImage("/Users/soswow/Pictures/Downloaded Albums/t992514/devclub-2011.02.25/IMG_7324.CR2.jpg")

    img = scale_image(normalize(img, aggressive=0.005))
    mask, seqs, time = get_mask_with_contour(img, ret_cont=True, ret_img=True, with_init_mask=False, time_took=True)
    boxes, min_rects = get_skin_rectangles(seqs,minsize=15)
    draw_boxes(boxes, img)
    return img, boxes, min_rects

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

    clusters = {}
    for i, tree in enumerate(cluster_forest):
        for node in tree.nodes():
            clusters[node] = i

    return clusters, mst


def main():
    img, boxes, min_rects = get_image_w_boxes()

    verticies = [(x+w/2, y+h/2) for x,y,w,h in boxes]
#    verticies = [(c[0],c[1]) for c,_,_ in min_rects]

#    for x,y,w,h in boxes:
#        verticies+=[(x,y), (x+w,y),(x,y+h),(x+w,y+h)]

    verticies = np.array(verticies)

    get_hemst_clusters(verticies, k=3)

    clusters, mst = draw_graph(img, mst, verticies, color=cv.RGB(255,0,255), thickness=2)

    colors = [cv.RGB(255,10,10),
             cv.RGB(255,255,10),
             cv.RGB(10,255,255),
             cv.RGB(255,10,255),
             cv.RGB(255,100,100),
             cv.RGB(255,255,100),
             cv.RGB(100,255,255),
             cv.RGB(255,100,255)]

    for i,xy in enumerate(verticies):
        cv.Circle(img, tuple(xy), 5, colors[clusters[i]], thickness=-1)

    show_image(img)

if __name__ == "__main__":
    main()
  