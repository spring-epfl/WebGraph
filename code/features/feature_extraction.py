import pandas as pd
import networkx as nx
from .dataflow import pre_extraction, get_dataflow_features
from .structure import get_structure_features

def extract_graph_node_features(G, df_graph, G_indirect, df_indirect_graph, G_indirect_all,
    node, dict_redirect, ldb, selected_features, vid):

    """
    Function to extract features for a node of the graph.

    Args:
      G: networkX representation of graph
      df_graph: DataFrame representation of graph
      G_indirect: networkX representation of shared information edges (indirect edges)
      df_indirect_graph: DataFrame representation of shared information edges
      G_indirect_all: networkX representation of direct and indirect edges
      node: URL of node whose features are extracted
      dict_redirect: dictionary of redirect depths of every node in graph
      ldb: Content LDB
      selected_features: features to extract
      vid: Visit ID
    Returns:
      df: DataFrame of features for the node.
    """
    

    all_features = []
    all_feature_names = ['visit_id', 'name']
    content_features = []
    structure_features = []
    dataflow_features = []
    additional_features = []
    content_feature_names = []
    structure_feature_names = []
    dataflow_feature_names = []
    additional_feature_names = []

    if 'content' in selected_features:
        content_features, content_feature_names = get_content_features(G, df_graph, node)
    if 'structure' in selected_features:
        structure_features, structure_feature_names = get_structure_features(G, df_graph, node, ldb)
    if 'dataflow' in selected_features:
        dataflow_features, dataflow_feature_names = get_dataflow_features(G, df_graph, node, dict_redirect, G_indirect, G_indirect_all, df_indirect_graph)
    if 'additional' in selected_features:
        additional_features, additional_feature_names = get_additional_features(G, df_graph, node)

    all_features = content_features + structure_features + dataflow_features + additional_features
    all_feature_names += content_feature_names + structure_feature_names + dataflow_feature_names + additional_feature_names

    df = pd.DataFrame([[vid] + [node] + all_features], columns=all_feature_names)

    return df

def extract_graph_features(df_graph, G, vid, ldb, feature_config):

    """
    Function to extract features.

    Args:
      df_graph: DataFrame of nodes/edges for a site
      G: networkX graph of site
      vid: Visit ID
      ldb: Content LDB
      feature_config: Feature config
    Returns:
      df_features: DataFrame of features for each URL in the graph

    This functions does the following:

    1. Reads the feature config to see which features we want.
    2. Creates a graph of indirect edges if we want to calculate dataflow features.
    3. Performs feature extraction based on the feature config. Feature extraction is per node of graph.
    """

    df_features = pd.DataFrame()
    nodes = G.nodes(data=True)
    G_indirect = nx.DiGraph()
    G_indirect_all = nx.DiGraph()
    df_indirect_graph = pd.DataFrame()
    dict_redirect = {}

    selected_features = feature_config['features_to_extract']
    if 'dataflow' in selected_features:
        dict_redirect, G_indirect, G_indirect_all, df_indirect_graph = pre_extraction(G, df_graph)

    for node in nodes:
        #Currently, we filter out Element and Storage nodes since we only want to classify URLs (the other nodes are used for feature calculation for these nodes though)
        if ("type" in node[1]) and (node[1]["type"] != "Element") and (node[1]['attr'] != "inline") and (node[1]["type"] != "Storage"):
            df_feature = extract_graph_node_features(
                G,
                df_graph,
                G_indirect,
                df_indirect_graph,
                G_indirect_all,
                node[0],
                dict_redirect,
                ldb,
                selected_features,
                vid
            )
            df_features = pd.concat([df_features, df_feature])
    
    return df_features
