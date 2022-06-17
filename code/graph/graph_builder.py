import json

import networkx as nx
import numpy as np


def build_networkx_graph(pdf):
    """
    Function to build a networkX graph from a Pandas DataFrame.

    Args:
        pdf: DataFrame of nodes and edges.
    Returns:
        G: networkX graph.

    This functions does the following:

    1. Selects nodes and edges.
    2. Processes node attributes.
    3. Creates graph from edges.
    4. Updates node attributes in graph.
    """

    df_nodes = pdf[(pdf["graph_attr"] == "Node") | (pdf["graph_attr"] == "NodeWG")]
    df_edges = pdf[(pdf["graph_attr"] == "Edge") | (pdf["graph_attr"] == "EdgeWG")]
    df_nodes = df_nodes.groupby(
        ['visit_id', 'name'],
        as_index=False
    ).agg(
        {
            'type': lambda x: list(x),
            'attr': lambda x: list(x),
            'domain' : lambda x: list(x)[0],
            'top_level_domain' : lambda x: list(x)[0]
        }
    )

    def modify_type(orig_type) -> str:
        orig_type = list(set(orig_type))
        if len(orig_type) == 1:
            return orig_type[0]

        new_type = "Request"
        if "Script" in orig_type:
            new_type = "Script"
        elif "Document" in orig_type:
            new_type = "Document"
        elif "Element" in orig_type:
            new_type = "Element"
        return new_type


    def modify_attr(orig_attr):
        orig_attr = np.array(list(set(orig_attr)))
        if len(orig_attr) == 1:
            return orig_attr[0]

        for item in orig_attr:
            if item and 'top_level_url' in item:
                return json.loads(item)
        return ""


    df_nodes['type'] = df_nodes['type'].apply(modify_type)
    df_nodes['attr'] = df_nodes['attr'].apply(modify_attr)
    networkx_graph = nx.from_pandas_edgelist(df_edges, source='src', target='dst', edge_attr=True, create_using=nx.DiGraph())
    node_dict = df_nodes.set_index('name').to_dict("index")
    nx.set_node_attributes(networkx_graph, node_dict)

    return networkx_graph
