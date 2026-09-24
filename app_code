import streamlit as st
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# Graph, Use Case: Emergency Supply Robot

locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6)
}

hospital_graph = {
    "Pharmacy": {
        "Main_Corridor": 2.2,
        "Patient_Wing": 4.1
    },

    "Main_Corridor": {
        "Nursing_Station": 2.2
    },

    "Patient_Wing": {
        "Laboratory": 5.0
    },

    "Nursing_Station": {
        "Laboratory": 3.2,
        "Emergency_Ward": 6.0
    },

    "Laboratory": {
        "Emergency_Ward": 3.2
    },

    "Emergency_Ward": {}
}


# Heuristic
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Path reconstruction
def reconstruct_path(came_from, current):
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# GBFS
def gbfs(start, goal):
    counter = 0
    frontier = [(heuristic(start, goal), counter, start)]
    came_from = {start: None}
    visited = set()

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = reconstruct_path(came_from, goal)
            cost = sum(hospital_graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
            return path, cost

        for neighbor in hospital_graph.get(current, {}):
            if neighbor not in visited and neighbor not in came_from:
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(frontier, (heuristic(neighbor, goal), counter, neighbor))

    return None, None


# A*
def a_star(start, goal):
    counter = 0
    g_cost = {start: 0}
    came_from = {start: None}
    frontier = [(heuristic(start, goal), counter, start)]
    visited = set()

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = reconstruct_path(came_from, goal)
            return path, g_cost[goal]

        for neighbor, cost in hospital_graph.get(current, {}).items():
            tentative_g = g_cost[current] + cost

            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(frontier, (f_score, counter, neighbor))

    return None, None


##########################################
# Streamlit GUI Code

# Set Page Config
st.set_page_config(page_title="Emergency Supply Robot - Search Visualizer", layout="centered")

# write meaningful title and description for the app
st.title("🏥 Emergency Supply Robot: Informed Search Visualizer")


# define the nodes and their coordinates
nodes = list(hospital_graph.keys())

# create a selectbox for the user to choose the start and goal nodes
start = st.selectbox(
    "Select Initial Node",
    nodes,
    index=nodes.index("Pharmacy")
)

goal = st.selectbox(
    "Select Goal Node",
    nodes,
    index=nodes.index("Emergency_Ward")
)

# create a selectbox for the user to choose the search algorithm
algorithm = st.selectbox(
    "Select Search Algorithm",
    ["GBFS", "A*"]
)


if st.button("Run Search"):

    if algorithm == "GBFS":

        # run the GBFS algorithm with the selected start and goal nodes
        path, cost = gbfs(start, goal)
    else:

        # run the A* algorithm with the selected start and goal nodes
        path, cost = a_star(start, goal)

    if path is None:

        # display a error message indicating that no path was found
        st.error(f"No path found between {start} and {goal}.")

    else:

        # Display result
        st.subheader("Search Result")

        st.write(
            f"Algorithm: {algorithm}"
        )

        st.write(
            f"Solution Path: {' → '.join(path)}"
        )

        st.write(
            f"Total Path Cost: {cost:.2f}"
        )

        # Visualize NetworkX graph

        G = nx.DiGraph()

        for node, neighbors in hospital_graph.items():

            for neighbor, weight in neighbors.items():

                G.add_edge(node, neighbor, weight=weight)

        pos = locations

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        # WRITE REMAINING NETWORKX VISUALIZATION CODE HERE
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1600, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
        nx.draw_networkx_edges(
            G, pos, edge_color="gray", arrows=True, arrowsize=20,
            connectionstyle="arc3,rad=0.08", ax=ax
        )
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(
            G, pos, edgelist=path_edges, edge_color="red", width=3,
            arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.08", ax=ax
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=path, node_color="orange", node_size=1600, ax=ax
        )

        ax.set_title(
            f"{algorithm} Solution Path"
        )

        ax.axis("off")

        st.pyplot(fig)
