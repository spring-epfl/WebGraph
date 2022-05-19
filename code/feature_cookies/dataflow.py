import numpy as np

import networkx as nx

from .utils import find_indirect_edges


def get_storage_features(df_graph, node):
    cookie_get = df_graph[
        (df_graph['dst'] == node) &
        ((df_graph['action'] == 'get') | (df_graph['action'] == 'get_js'))
    ]

    cookie_set = df_graph[
        (df_graph['dst'] == node) &
        ((df_graph['action'] == 'set') | (df_graph['action'] == 'set_js'))
    ]

    localstorage_get = df_graph[
        (df_graph['dst'] == node) &
        (df_graph['action'] == 'get_storage_js')
    ]

    localstorage_set = df_graph[
        (df_graph['dst'] == node) &
        (df_graph['action'] == 'set_storage_js')
    ]

    num_get_storage = len(cookie_get) + len(localstorage_get)
    num_set_storage = len(cookie_set) + len(localstorage_set)

    storage_features = [num_get_storage, num_set_storage]
    storage_feature_names = ['num_get_storage', 'num_set_storage']

    return storage_features, storage_feature_names


def get_request_flow_features(G, df_graph, node):
    #Request flow features
    predecessors = list(G.predecessors(node))
    predecessors_type = [G.nodes[x].get('type') for x in predecessors]
    num_script_predecessors = len([x for x in predecessors_type if x == "Script"])

    rf_features = [num_script_predecessors]

    rf_feature_names = ['num_script_predecessors']

    return rf_features, rf_feature_names


def get_indirect_features(G, df_graph, node):
    """
    Function to extract indirect edge features (specified in features.yaml file)

    Args:
      node: URL of node
      G: networkX graph of indirect edges
      df_graph: DataFrame representation of graph (indirect edges only)
    Returns:
      List of indirect edge features
    """

    in_degree = -1
    out_degree = -1
    ancestors = -1
    descendants = -1
    closeness_centrality = -1
    average_degree_connectivity = -1
    eccentricity = -1
    mean_in_weights = -1
    min_in_weights = -1
    max_in_weights = -1
    mean_out_weights = -1
    min_out_weights = -1
    max_out_weights = -1
    num_set_get_src = 0
    num_set_mod_src = 0
    num_set_url_src = 0
    num_get_url_src = 0
    num_set_get_dst = 0
    num_set_mod_dst = 0
    num_set_url_dst = 0
    num_get_url_dst = 0

    try:
        if len(df_graph) > 0:
            num_set_get_src = len(df_graph[(df_graph['type'] == 'set_get') & (df_graph['src'] == node)])
            num_set_mod_src = len(df_graph[(df_graph['type'] == 'set_modify') & (df_graph['src'] == node)])
            num_set_url_src = len(df_graph[(df_graph['type'] == 'set_url') & (df_graph['src'] == node)])
            num_get_url_src = len(df_graph[(df_graph['type'] == 'get_url') & (df_graph['src'] == node)])
            num_set_get_dst = len(df_graph[(df_graph['type'] == 'set_get') & (df_graph['dst'] == node)])
            num_set_mod_dst = len(df_graph[(df_graph['type'] == 'set_modify') & (df_graph['dst'] == node)])
            num_set_url_dst = len(df_graph[(df_graph['type'] == 'set_url') & (df_graph['dst'] == node)])
            num_get_url_dst = len(df_graph[(df_graph['type'] == 'get_url') & (df_graph['dst'] == node)])

        if G != "Empty" and node in G.nodes():
            in_degree = G.in_degree(node)
            out_degree = G.in_degree(node)
            ancestors = len(nx.ancestors(G, node))
            descendants = len(nx.descendants(G, node))
            closeness_centrality = nx.closeness_centrality(G, node)
            average_degree_connectivity = [*nx.average_degree_connectivity(G, nodes=[node]).values()][0]
            try:
                H = G.copy().to_undirected()
                eccentricity = nx.eccentricity(H, node)
            except Exception as e:
                eccentricity = -1
            in_weights = df_graph[(df_graph['dst'] == node)]['attr'].tolist()
            out_weights = df_graph[(df_graph['src'] == node)]['attr'].tolist()

            if len(in_weights) > 0:
                mean_in_weights = np.mean(in_weights)
                min_in_weights = min(in_weights)
                max_in_weights = max(in_weights)

            if len(out_weights) > 0:
                mean_out_weights = np.mean(out_weights)
                min_out_weights = min(out_weights)
                max_out_weights = max(out_weights)
    except Exception as e:
        print(e)

    indirect_features = [
        in_degree, out_degree, ancestors, descendants, closeness_centrality,
        average_degree_connectivity, eccentricity, mean_in_weights,
        min_in_weights, max_in_weights, mean_out_weights, min_out_weights,
        max_out_weights, num_set_get_src, num_set_mod_src, num_set_url_src,
        num_get_url_src, num_set_get_dst, num_set_mod_dst, num_set_url_dst,
        num_get_url_dst
    ]

    indirect_feature_names = [
        'indirect_in_degree', 'indirect_out_degree', 'indirect_ancestors', 'indirect_descendants', 'indirect_closeness_centrality',
        'indirect_average_degree_connectivity', 'indirect_eccentricity', 'indirect_mean_in_weights',
        'indirect_min_in_weights', 'indirect_max_in_weights', 'indirect_mean_out_weights', 'indirect_min_out_weights',
        'indirect_max_out_weights', 'num_set_get_src', 'num_set_mod_src', 'num_set_url_src',
        'num_get_url_src', 'num_set_get_dst', 'num_set_mod_dst', 'num_set_url_dst',
        'num_get_url_dst'
    ]

    return indirect_features, indirect_feature_names


def get_indirect_all_features(G, node):
    """
    Function to extract all indirect edge features (specified in features.yaml file)

    Args:
      node: URL of node
      G: networkX graph (of both direct and indirect edges)
    Returns:
      List of all indirect features
    """

    in_degree = -1
    out_degree = -1
    ancestors = -1
    descendants = -1
    closeness_centrality = -1
    average_degree_connectivity = -1
    eccentricity = -1

    try:
        if G != "Empty" and node in G.nodes():
            in_degree = G.in_degree(node)
            out_degree = G.in_degree(node)
            ancestors = len(nx.ancestors(G, node))
            descendants = len(nx.descendants(G, node))
            closeness_centrality = nx.closeness_centrality(G, node)
            average_degree_connectivity = [*nx.average_degree_connectivity(G, nodes=[node]).values()][0]
            try:
                H = G.copy().to_undirected()
                eccentricity = nx.eccentricity(H, node)
            except Exception as e:
                eccentricity = -1
    except Exception as e:
        print(e)

    indirect_all_features = [
        in_degree, out_degree, ancestors, descendants,
        closeness_centrality, average_degree_connectivity, eccentricity
    ]
    indirect_all_feature_names = [
        'indirect_all_in_degree', 'indirect_all_out_degree',
        'indirect_all_ancestors', 'indirect_all_descendants',
        'indirect_all_closeness_centrality',
        'indirect_all_average_degree_connectivity',
        'indirect_all_eccentricity'
    ]

    return indirect_all_features, indirect_all_feature_names


def get_dataflow_features(G, df_graph, node, G_indirect, G_indirect_all, df_indirect_graph):
    all_features = []
    all_feature_names = []

    storage_features, storage_feature_names = get_storage_features(df_graph, node)
    rf_features, rf_feature_names = get_request_flow_features(G, df_graph, node)
    indirect_features, indirect_feature_names =  get_indirect_features(G_indirect, df_indirect_graph, node)
    indirect_all_features, indirect_all_feature_names = get_indirect_all_features(G_indirect_all, node)

    all_features = storage_features + rf_features + indirect_features + indirect_all_features
    all_feature_names = storage_feature_names + rf_feature_names + indirect_feature_names + indirect_all_feature_names

    return all_features, all_feature_names


def pre_extraction(G, df_graph):
    G_indirect, df_indirect_graph = find_indirect_edges(G, df_graph)

    if G_indirect != "Empty":
        G_indirect_all = nx.compose(G, G_indirect)

    else:
        G_indirect_all = "Empty"

    return G_indirect, G_indirect_all, df_indirect_graph

