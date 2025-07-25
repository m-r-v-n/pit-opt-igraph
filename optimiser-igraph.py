############################################################
# igraph Pit Optimiser - Goldberg-Tarjan Algorithm (Push-Relabel)
#
# 
# 
# 
#
############################################################

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import sin, cos, tan, radians
from igraph import plot
import igraph as ig
from igraph import Graph
import os

# Sending nodes to Sink or Source

def sendNodes(BM, sink, source, g, BVal):
    Sink = sink
    Source = source
    rows = sink - 1
    start_UPL = time.time()

    edges = []
    capacities = []

    for i in range(rows):
        node = i + 1
        capacity = np.abs(np.round(BM[i, BVal], 2))

        if BM[i, BVal] < 0:
            edges.append((node, Sink))
            capacities.append(capacity)
        else:
            edges.append((Source, node))
            capacities.append(capacity)

    g.add_edges(edges)
    g.es[-len(edges):]["capacity"] = capacities

    print(f"Total vertices: {g.vcount()}")
    used_nodes = [e.source for e in g.es] + [e.target for e in g.es]
    print(f"Max node ID used: {max(used_nodes)}")

    print("--> Create external arc time: --%s seconds " % (np.round((time.time() - start_UPL), 2)))
    return g

def createArcPrecedence(BM,
                        idx,
                        xsize,ysize,zsize,
                        xmin,ymin,zmin,
                        xmax,ymax,zmax,
                        xcol,ycol,zcol,
                        slopecol,
                        num_blocks_above,
                        g,
                        minWidth):

    start_UPL = time.time()

    BM1 = BM[:, [idx, xcol, ycol, zcol]]
    block_to_value = {(x, y, z): value for value, x, y, z in BM1}

    edges = []
    capacities = []
    internal_arc = 0
    nodes_with_edge = 0

    BM2 = BM[:, [xcol, ycol, zcol, slopecol]]

    for i, (x_i, y_i, z_i, angle_i) in enumerate(BM2, start=1):
        min_radius = minWidth / 2

        if z_i == zmax and min_radius == 0:
            continue

        cone_height = zsize * num_blocks_above
        cone_radius = cone_height / math.tan(math.radians(angle_i))

        search_x = int(np.ceil((cone_radius - (xsize / 2)) / xsize))
        search_y = int(np.ceil((cone_radius - (ysize / 2)) / ysize))

        x_range = range(-int(min(((x_i - xmin) / xsize), search_x)),
                        int(min(((xmax - x_i) / xsize) + 1, search_x + 1)))
        y_range = range(-int(min(((y_i - ymin) / ysize), search_y)),
                        int(min(((ymax - y_i) / ysize) + 1, search_y + 1)))
        z_range = range(0, int(min(((zmax - z_i) / zsize) + 1, num_blocks_above + 1)))

        block_coords = np.array([
            (x_i + j * xsize, y_i + k * ysize, z_i + l * zsize)
            for j in x_range
            for k in y_range
            for l in z_range
        ])

        if block_coords.size == 0:
            continue

        dists = np.sqrt((block_coords[:, 0] - x_i) ** 2 + (block_coords[:, 1] - y_i) ** 2)
        heights = block_coords[:, 2] - z_i

        with np.errstate(divide='ignore', invalid='ignore'):
            cone_radii = (heights * cone_radius / cone_height)

        if min_radius > 0:
            cone_radii = np.where(heights == 0, minWidth, cone_radii + min_radius)

        inside_indices = np.where(dists <= cone_radii)[0]
        inside_blocks = block_coords[inside_indices]

        connected = 0
        for block in inside_blocks:
            block_key = tuple(block)
            source_key = (x_i, y_i, z_i)

            if block_key not in block_to_value or source_key not in block_to_value:
                continue

            target = int(block_to_value[block_key])
            source = int(block_to_value[source_key])

            if source == target:
                continue

            edges.append((source, target))
            capacities.append(99e99)
            connected += 1
            internal_arc += 1

        if connected > 0:
            nodes_with_edge += 1
            arc_rate = np.around(internal_arc / (time.time() - start_UPL), 2)
            #print(f"index = {i} node = {source} connected arcs = {connected} total arcs = {internal_arc} arc rate = {arc_rate}")
            print(f"index = {i} node = {source} connected arcs = {connected} total arcs generated = {internal_arc} x = {x_i} y = {y_i} z = {z_i} angle = {angle_i} arc gen rate = {arc_rate}")

    print("Block precedence search complete")
    print("Adding edges to the graph")
    g.add_edges(edges)
    g.es[-len(edges):]["capacity"] = capacities

    total_int_arc_rate = np.around(internal_arc / (time.time() - start_UPL), 2)
    #total_node_rate = np.around(nodes_with_edge / (time.time() - start_UPL), 2)

    print("\nPerformance:")
    print(f"--- Total Nodes Processed: {i}")
    print(f"--- Nodes with Edges: {nodes_with_edge}")
    print(f"--- Total Precedence Arcs: {internal_arc}")
    print(f"--- Average Arcs per Node: {np.around(internal_arc / nodes_with_edge, 0)} arcs/node")
    print(f"--- Precedence Arc Generation Rate: {total_int_arc_rate}/s")
    print("--> Precedence Arc Generation time: %s seconds" % (np.round(time.time() - start_UPL, 2)))

    return g

def iGraphMF_UPL(BM,
                   sink,
                   source,
                   idx,
                   xsize,ysize,zsize,
                   xmin,ymin,zmin,
                   xmax,ymax,zmax,
                   xcol,ycol,zcol,
                   slopecol,
                   num_blocks_above,
                   BVal,
                   pitLimit,
                   Cashflow,
                   minWidth):

    print("Process Start...")
    start_UPL = time.time()

    x_coords = BM[:, xcol]
    y_coords = BM[:, ycol]
    z_coords = BM[:, zcol]

    num_vertices = sink + 1
    g = Graph(directed=True)
    g.add_vertices(num_vertices)

    g.vs["x"] = x_coords.tolist()
    g.vs["y"] = y_coords.tolist()
    g.vs["z"] = z_coords.tolist()

    print("Sending Nodes")
    g = sendNodes(BM, sink, source, g, BVal)
    print("External Arcs done")

    print("Creating Precedence")
    g = createArcPrecedence(BM,
                            idx,
                            xsize,ysize,zsize,
                            xmin,ymin,zmin,
                            xmax,ymax,zmax,
                            xcol,ycol,zcol,
                            slopecol,
                            num_blocks_above,
                            g,
                            minWidth)

    print("Solving Ultimate Pit Limit")
    solve_UPL = time.time()

    flow = g.maxflow(source, sink, capacity="capacity")
    mincut_vertices = flow.partition
    cut_set = set(mincut_vertices[0])
    InsideList = [v for v in cut_set if v != source and v != sink]

    BM[:, pitLimit] = 0
    BM[:, Cashflow] = 0

    for indUPL in InsideList:
        BM[int(indUPL) - 1, pitLimit] = 1
        BM[int(indUPL) - 1, Cashflow] = BM[int(indUPL) - 1, BVal]

    cashFlow = "{:,.2f}".format(np.sum(BM[:, Cashflow]))
    print("--> iGraph Maxflow Optimization time: --%s seconds " % (np.around((time.time() - solve_UPL), decimals=2)))
    print("--> Total process time: --%s seconds " % (np.around((time.time() - start_UPL), decimals=2)))
    print(f"Undiscounted Cashflow: ${cashFlow}")

    return BM

def main():
    print("Start")
    start_time = time.time()

########################################################################################

    # 1. Block model location
    filePath = 'marvin_copper_final.csv'

    # 2. Block model size
    xsize = 30
    ysize = 30
    zsize = 30

    # 3. Column number of xyz coordinates. note that column number starts at 0
    xcol = 1
    ycol = 2
    zcol = 3

    # 4. Block search boundary parameters
    num_blocks_above = 6

    # 5. Minimum mining width for pit bottom consideration (this will be added to the radius of the search cone)
    minWidth = 0.0

    # 6. Column number of Block Value (column number starts at 0)
    BVal = 7 

    # 7. Column numbler of Slope Angle
    slopecol = 8

    # 8. Column numbler of Index ID
    idx = 0

########################################################################################

    data = np.loadtxt(filePath, delimiter=',', skiprows=1) # Import Block Model

    x_col = data[:, xcol]
    y_col = data[:, ycol]
    z_col = data[:, zcol]

    xmin = x_col.min()
    xmax = x_col.max()

    ymin = y_col.min()
    ymax = y_col.max()

    zmin = z_col.min()
    zmax = z_col.max()

    nx = ((xmax - xmin) / xsize) + 1
    ny = ((ymax - ymin) / ysize) + 1
    nz = ((zmax - zmin) / zsize) + 1

    sink = int((data.shape[0]) + 1)
    source = 0

    orig_col = data.shape[1]
    print(f"Original Column = {orig_col}")

    # Add two new columns for pitLimit and CashFlow
    n_rows = data.shape[0]
    col1 = np.zeros((n_rows, 1))
    col2 = np.zeros((n_rows, 1))
    data = np.hstack((data, col1, col2))

    # Store column numbers (indices) of new columns
    pitLimit = orig_col      # first new column
    CashFlow = orig_col + 1  # second new column
    print(f"UPL Column = {pitLimit}")
    print(f"Cashflow Column = {CashFlow}")

    BlockModel = data

    # Call iGraph Maxflow function
    BlockModel = iGraphMF_UPL(BlockModel,
                                sink,
                                source,
                                idx,
                                xsize,ysize,zsize,
                                xmin,ymin,zmin,
                                xmax,ymax,zmax,
                                xcol,ycol,zcol,
                                slopecol,
                                num_blocks_above,
                                BVal,
                                pitLimit,
                                CashFlow,
                                minWidth
                                )

    # Save Block Model
    base, ext = os.path.splitext(filePath)

    np.savetxt(
        fname=f"{base}_opt{ext}",
        X=BlockModel,
        fmt='%.3f',
        delimiter=',',
        header="id+1,X,Y,Z,tonne,au_ppm,cu_pct,block_val,slope,pit_limit,cash_flow",
        comments=''  # <- this removes the default '#' comment character
    )
    
if __name__ == "__main__":
    main()    
