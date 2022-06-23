import networkx as nx
import re
from .utils import *
import time

from logger import LOGGER

def get_script_content_features(G, df_graph, node, ldb):

  """
  Function to extract script content features.

  Args:
    node: URL of node
    G: networkX graph
    df_graph: DataFrame representation of graph
    ldb: content LDB
  Returns:
    sc_features: script content feature values
    sc_features_names: script content feature names
  """

  keywords_fp = ["CanvasRenderingContext2D", "HTMLCanvasElement", "toDataURL",
                  "getImageData", "measureText", "font", "fillText", "strokeText",
                  "fillStyle", "strokeStyle", "HTMLCanvasElement.addEventListener",
                  "save", "restore"]
  ancestors = nx.ancestors(G, node)
  ascendant_script_has_eval_or_function = 0
  ascendant_script_has_fp_keyword = 0
  ascendant_script_length = 0
  max_length = 0

  try:
    for ancestor in ancestors:
      try:
        if nx.get_node_attributes(G, 'type')[ancestor] == 'Script':
          content_hash = df_graph[(df_graph['dst'] == ancestor) & (df_graph['content_hash'] != "N/A") & (df_graph['content_hash'].notnull())]["content_hash"]
          if len(content_hash) > 0:
            ldb_key = content_hash.iloc[0].encode('utf-8')
            script_content = ldb.Get(ldb_key)
            if len(script_content) > 0:
              script_content = script_content.decode('utf-8')
              script_length = len(script_content)
              if script_length > max_length:
                ascendant_script_length = script_length
                max_length = script_length
              if ('eval' in script_content) or ('function' in script_content):
                ascendant_script_has_eval_or_function = 1
              for keyword in keywords_fp:
                if keyword in script_content:
                  ascendant_script_has_fp_keyword = 1
                  break
      except Exception as e:
        continue
  except Exception as e:
    LOGGER.warning("[ get_script_content_features ] : ERROR - ", exc_info=True)

  sc_features = [ascendant_script_has_eval_or_function, ascendant_script_has_fp_keyword, \
              ascendant_script_length]
  sc_features_names = ['ascendant_script_has_eval_or_function', 'ascendant_script_has_fp_keyword', \
              'ascendant_script_length']

  return sc_features, sc_features_names


def get_ne_features(G):

  """
  Function to extract graph features.

  Args:
    G: networkX graph
  Returns:
    ne_features: graph feature values
    ne_feature_names: graph feature names
  """

  num_nodes = len(G.nodes)
  num_edges = len(G.edges)
  num_nodes_div = num_nodes
  num_edges_div = num_edges
  if num_edges == 0:
    num_edges_div = 0.000001
  if num_nodes == 0:
    num_nodes_div = 0.000001
  nodes_div_by_edges = num_nodes/num_edges_div
  edges_div_by_nodes = num_edges/num_nodes_div
  ne_features = [num_nodes, num_edges, nodes_div_by_edges, edges_div_by_nodes]
  ne_feature_names = ['num_nodes', 'num_edges', 'nodes_div_by_edges', 'edges_div_by_nodes']

  return ne_features, ne_feature_names

def get_connectivity_features(G, df_graph, node):

  """
  Function to extract connectivity features.

  Args:
    G: networkX graph
    df_graph: DataFrame representation of graph
    node: URL of node
  Returns:
    connectivity_features: connectivity feature values
    connectivity_feature_names: connectivity feature names
  """

  connectivity_features = []

  in_degree = 0
  out_degree = 0
  in_out_degree = 0
  ancestors = 0
  descendants = 0
  closeness_centrality = 0
  average_degree_connectivity = 0
  eccentricity = 0
  ascendant_has_ad_keyword = 0
  is_eval_or_function = 0

  is_parent_script = 0
  is_ancestor_script = 0
  descendant_of_eval_or_function = 0

  try:

    in_degree = G.in_degree(node)
    out_degree = G.out_degree(node)
    in_out_degree = in_degree + out_degree
    ancestor_list = nx.ancestors(G, node)
    ancestors = len(ancestor_list)
    descendants = len(nx.descendants(G, node))
    parents = list(G.predecessors(node))
    is_eval_or_function = 0

    path_lengths = nx.single_source_shortest_path_length(G, node)

    for parent in parents:
      try:
        if nx.get_node_attributes(G, 'type')[parent] == 'Script':
          is_parent_script = 1
        if nx.get_node_attributes(G, 'type')[parent] == 'Element':
          attr = nx.get_node_attributes(G, 'attr')[parent]
          attr = json.loads(attr)
          if (attr['eval']) and (attr['subtype'] == 'script'):
            is_eval_or_function = 1
        if is_parent_script & is_eval_or_function:
          break
      except:
        continue

    for ancestor in ancestor_list:
      try:
        if nx.get_node_attributes(G, 'type')[ancestor] == 'Script':
          is_ancestor_script = 1
        if nx.get_node_attributes(G, 'type')[ancestor] == 'Element':
          attr = nx.get_node_attributes(G, 'attr')[ancestor]
          attr = json.loads(attr)
          if (attr['eval']) and (attr['subtype'] == 'script'):
            descendant_of_eval_or_function = 1
        if is_ancestor_script & descendant_of_eval_or_function:
          break
      except:
        continue

    parent_flag = 0
    descendant_flag = 0


    ascendant_has_ad_keyword = ad_keyword_ascendants(node, G)
    closeness_centrality = nx.closeness_centrality(G, node)
    average_degree_connectivity = nx.average_degree_connectivity(G)[in_out_degree]

    try:
      H = G.copy().to_undirected()
      eccentricity = nx.eccentricity(H, node)
    except Exception as e:
      eccentricity = -1

    connectivity_features = [in_degree, out_degree, in_out_degree, \
          ancestors, descendants, closeness_centrality, average_degree_connectivity, \
          eccentricity, is_parent_script, is_ancestor_script, \
          ascendant_has_ad_keyword, is_eval_or_function, \
          descendant_of_eval_or_function]

  except Exception as e:
    LOGGER.warning("Error in connectivity features:", exc_info=True)
    connectivity_features = [in_degree, out_degree, in_out_degree, \
          ancestors, descendants, closeness_centrality, average_degree_connectivity, \
          eccentricity, is_parent_script, is_ancestor_script, \
          ascendant_has_ad_keyword, is_eval_or_function, \
          descendant_of_eval_or_function]

  connectivity_feature_names = ['in_degree', 'out_degree', 'in_out_degree', \
          'ancestors', 'descendants', 'closeness_centrality', 'average_degree_connectivity', \
          'eccentricity', 'is_parent_script', 'is_ancestor_script', \
          'ascendant_has_ad_keyword', 'is_eval_or_function', \
          'descendant_of_eval_or_function']

  return connectivity_features, connectivity_feature_names


def get_structure_features(G, df_graph, node, ldb):

  """
  Function to extract structural features. This function calls
  the other functions to extract different types of structural features.

  Args:
    G: networkX graph
    df_graph: DataFrame representation of graph
    node: URL of node
    ldb: content LDB
  Returns:
    all_features: strcuture feature values
    all_feature_names: structure feature names
  """

  all_features = []
  all_feature_names = []
  ne_features, ne_feature_names = get_ne_features(G)
  connectivity_features, connectivity_feature_names = get_connectivity_features(G, df_graph, node)
  sc_features, sc_features_names = get_script_content_features(G, df_graph, node, ldb)

  all_features = ne_features + connectivity_features + sc_features
  all_feature_names = ne_feature_names + connectivity_feature_names + sc_features_names

  return all_features, all_feature_names

