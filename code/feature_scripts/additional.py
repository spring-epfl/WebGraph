import pandas as pd
import networkx as nx
import json
import numpy as np

def get_dataflow_additional(df_graph, node):

  created_elements = df_graph[(df_graph['src'] == node) & \
                    (df_graph['action'] == 'create')]
  num_created_elements = len(created_elements)
  return [num_created_elements], ['num_created_elements']

def get_structure_additional(G, df_graph, node):

  num_cs_edges_sent = 0
  num_cs_edges_rec = 0
  num_diff_domain_predecessors = 0
  num_diff_domain_successors = 0
  num_diff_domain_ancestors = 0
  num_diff_domain_descendants = 0

  try:

    cs_edges_sent = df_graph[(df_graph['src'] == node) & ((df_graph['reqattr'] == "CS") | (df_graph['attr'] == "CS"))]
    cs_edges_rec = df_graph[(df_graph['dst'] == node) & ((df_graph['reqattr'] == "CS") | (df_graph['attr'] == "CS"))]
    num_cs_edges_sent = len(cs_edges_sent)
    num_cs_edges_rec = len(cs_edges_rec)

    predecessors = list(G.predecessors(node))
    successors = list(G.successors(node))
    ancestors = list(nx.ancestors(G, node))
    descendants = list(nx.descendants(G, node))

    predecessors_ps1 = [G.nodes[x]['domain'] for x in predecessors]
    successors_ps1 = [G.nodes[x]['domain'] for x in successors]
    ancestors_ps1 = [G.nodes[x]['domain'] for x in ancestors]
    descendants_ps1 = [G.nodes[x]['domain'] for x in descendants]
    node_ps1 = G.nodes[node]['domain']
    
    num_diff_domain_predecessors = len([x for x in predecessors_ps1 if x != node_ps1])
    num_diff_domain_successors = len([x for x in successors_ps1 if x != node_ps1])
    num_diff_domain_ancestors = len([x for x in ancestors_ps1 if x != node_ps1])
    num_diff_domain_descendants = len([x for x in descendants_ps1 if x != node_ps1])

    sa_features = [num_cs_edges_sent, num_cs_edges_rec, num_diff_domain_predecessors, \
      num_diff_domain_successors, num_diff_domain_ancestors, num_diff_domain_descendants]

  except:
    sa_features = [num_cs_edges_sent, num_cs_edges_rec, num_diff_domain_predecessors, \
      num_diff_domain_successors, num_diff_domain_ancestors, num_diff_domain_descendants]

  sa_feature_names = ['num_cs_edges_sent', 'num_cs_edges_rec', 'num_diff_domain_predecessors', \
      'num_diff_domain_successors', 'num_diff_domain_ancestors', 'num_diff_domain_descendants']
  
  return sa_features, sa_feature_names

def get_response_features(df_graph, node):

  rec_response_attr = df_graph[(df_graph['src'] == node) & (df_graph['attr'].notnull()) & (df_graph['attr'] != "CS") & (df_graph['respattr'] != "N/A") & (df_graph['attr'] != "N/A")]['attr'].tolist()
  rec_response_attr = [x for x in rec_response_attr if len(x) > 0]
  rec_response_lengths = [json.loads(x).get("clength") for x in rec_response_attr]
  rec_response_lengths = list(filter(None, rec_response_lengths))

  response_attr = df_graph[(df_graph['dst'] == node) & (df_graph['attr'].notnull()) & (df_graph['attr'] != "CS") & (df_graph['respattr'] != "N/A") & (df_graph['attr'] != "N/A")]['attr'].tolist()
  response_attr = [x for x in response_attr if len(x) > 0]
  response_lengths = [json.loads(x).get("clength") for x in response_attr]
  response_lengths = list(filter(None, response_lengths))

  max_rec_response_length = -1
  min_rec_response_length = -1
  mean_rec_response_length = -1
  if len(rec_response_lengths) > 0:
    max_rec_response_length = max(rec_response_lengths)
    min_rec_response_length = min(rec_response_lengths)
    mean_rec_response_length = np.mean(rec_response_lengths)

  max_size_response = -1
  min_size_response = -1
  mean_size_response = -1
  if len(response_lengths) > 0:
    max_size_response = max(response_lengths)
    min_size_response = min(response_lengths)
    mean_size_response = np.mean(response_lengths)

  response_features = [max_rec_response_length, min_rec_response_length, \
      mean_rec_response_length, max_size_response, min_size_response, \
      mean_size_response]

  response_feature_names = ['max_rec_response_length', 'min_rec_response_length', \
      'mean_rec_response_length', 'max_size_response', 'min_size_response', \
      'mean_size_response']

  return response_features, response_feature_names

def get_cookie_features(G, df_graph, node):

  """
  Function to extract cookie features (specified in features.yaml file)

  Args:
    node: URL of node
    G: networkX graph
    df_graph: DataFrame representation of graph
    top_level_url: Top level URL
  Returns:
    List of cookie features
  """

  max_size_name = -1
  max_size_val = -1
  min_size_name = -1
  min_size_val = -1
  mean_size_name = -1
  mean_size_val = -1
  num_httponly = 0
  num_diff_domain = 0

  cookie_set = df_graph[(df_graph['src'] == node) & ((df_graph['action'] == 'set') | (df_graph['action'] == 'set_js'))]
  cookie_set_elements = cookie_set['dst'].tolist()
  cookie_set_edge_attr = cookie_set['attr'].tolist()

  size_val = []
  size_name = []
  for item in cookie_set_edge_attr:
    try:
      if "N/A" not in item:
        attr = json.loads(item)
        size_name.append(len(attr['name']))
        val = attr['value']
        if 'none' in val.lower():
          size_val.append(0)
        else:
          size_val.append(len(val))
        if 'httponly' in attr and attr['httponly'] == True:
          num_httponly += 1
        if 'domain' in attr and attr['domain'] != None:
          top_level_domain = G.nodes[node]['top_level_domain']
          cookie_domain = attr['domain'][1:]
          if cookie_domain != top_level_domain:
            num_diff_domain += 1
    except Exception as e:
      print("Cookie features error", e)

  if len(size_name) > 0 and len(size_val) > 0:
    max_size_name = max(size_name)
    max_size_val =  max(size_val)
    min_size_name = min(size_name)
    min_size_val = min(size_val)
    mean_size_name = np.mean(size_name)
    mean_size_val = np.mean(size_val)

  cookie_features = [max_size_name, max_size_val, min_size_name, min_size_val, mean_size_name, \
    mean_size_val, num_httponly, num_diff_domain]
  cookie_feature_names = ['max_size_name', 'max_size_val', 'min_size_name', 'min_size_val', \
                          'mean_size_name', 'mean_size_val', 'num_httponly', 'num_diff_domain']

  return cookie_features, cookie_feature_names

def get_content_additional(G, df_graph, node):

  content_features = []
  cookie_features, cookie_feature_names = get_cookie_features(G, df_graph, node)
  response_features, response_feature_names = get_response_features(df_graph, node)

  content_features = cookie_features + response_features
  content_feature_names = cookie_feature_names + response_feature_names

  return content_features, content_feature_names

def get_additional_features(G, df_graph, node):

  all_features = []
  all_feature_names = []
  content_features, content_feature_names = get_content_additional(G, df_graph, node)
  structure_features, structure_feature_names = get_structure_additional(G, df_graph, node)
  data_flow_features, data_flow_feature_names = get_dataflow_additional(df_graph, node)
  all_features = content_features + structure_features + data_flow_features
  all_feature_names = content_feature_names + structure_feature_names + data_flow_feature_names

  return all_features, all_feature_names

