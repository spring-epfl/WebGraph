from context import *
from mutate_utils import *
from obfuscation import * 

import pandas as pd
import datetime
from networkx.readwrite import json_graph
import random
import logging
from logging.config import fileConfig

fileConfig('logging.conf')
logger = logging.getLogger('styles')


def add_node(current_node_f, df_graph_vid, G, third_party_node_names, original_tlu, ct, result_dir, visit_id, 
             ldb, feature_config, clf, mapping_dict, filterlists, filterlist_rules):

    """
    Function to return a mutated graph by node addition.

    Args:
        current_node_f: Node that is the parent to add a new child to.
        df_graph_vid: DataFrame representing graph
        G: Graph in networkx format
        third_party_node_names: List of adversary URLs
        original_tlu: Top level URL
        ct: Iteration number
        result_dir: Results folder path
        visit_id: Visit ID
        ldb: Content LDB
        feature_config: Feature config
        clf: Classifier
        mapping_dict: Dictionary of content mutation mappings
        filterlists: Names of filter lists to label nodes
        filterlist_rules: Filter list rules to label nodes.
    Returns:
        df: DataFrame of switches in classification 
        df_graph_new: DataFrame of new graph 
        new_node_f: Newly added node

    This functions does the following:

    1. Create a fake child node.
    2. Add the child and its edge to the graph.
    3. Perform classification.
    4. Compare with original classifications (original = content mutated graph)
    5. Return results.
    
    """

    G_copy = G.copy()
    current_node = current_node_f.iloc[0]
    new_node_f, new_edge_dict = create_child(current_node['type'], current_node['name'], current_node['visit_id'], original_tlu)
    
    new_node = new_node_f.iloc[0]
    new_edge_dict['src'] = current_node['name']
    new_edge_dict['dst'] = new_node['name']
    df_new_edge = pd.DataFrame({k:[v] for k, v in new_edge_dict.items()})
    df_graph_new = pd.concat([df_graph_vid, new_node_f, df_new_edge], axis=0, ignore_index=True)
   
    G_copy.add_node(new_node['name'], visit_id=int(visit_id), type=new_node['type'], attr=new_node['attr'])
    G_copy.add_edge(current_node['name'], new_node['name'], visit_id=int(new_edge_dict['visit_id']),\
        action=new_edge_dict['action'], reqattr=new_edge_dict['reqattr'],\
        respattr=new_edge_dict['respattr'], attr=new_edge_dict['attr'],\
        response_status=int(new_edge_dict['response_status']), time_stamp=new_edge_dict['time_stamp'])

    folder_tag = str(ct) + "_" + str(random.randint(0,10000)) + "_add_node"
    num_test = extract_and_classify(df_graph_new, G, result_dir, folder_tag, visit_id, ldb, feature_config, clf, filterlists, filterlist_rules)
    
    new_dirname = os.path.join(result_dir, folder_tag)
    #Uncomment if comparison is with original graph
    #df_original = read_pred(os.path.join(result_dir, "original", "tp_0"))
    #Comparison with content mutated graph
    df_original = read_prediction_df(os.path.join(result_dir, "0_content_mutated", "tp_0"))
    df_new = read_prediction_df(os.path.join(new_dirname, "tp_0"))
    result_dict, diff = calculate_misclassifications_mutated(df_new, df_original, mapping_dict, third_party_node_names, original_tlu)
    result_dict = json.dumps(result_dict)

    G_data = json_graph.node_link_data(G_copy)
    G_json = json.dumps(G_data)
    change_info = {}
    change_info['type'] = 'node_addition'
    change_info['info'] = new_node_f.to_dict('index')

    data = {'result_dict': [result_dict], 
            'diff': [diff], 
            'graph': [G_json], 
            'change_info': [change_info],
            'folder_tag': [folder_tag],
            }
    df = pd.DataFrame.from_dict(data)

    return df, df_graph_new, new_node_f

def find_storage_edges(df_graph_vid, third_party_node_names):

    df_node = df_graph_vid[(df_graph_vid['graph_attr'] == 'Node') | (df_graph_vid['graph_attr'] == 'NodeWG')]
    storage_node_names = df_node[(df_node['type'] == 'Storage')]['name'].unique().tolist()
    
    df_storage_edges = df_graph_vid[df_graph_vid['dst'].isin(storage_node_names)]
    df_storage_edges = df_storage_edges[(df_storage_edges['action'] == 'set') |
                                        (df_storage_edges['action'] == 'set_js')]
    df_storage_edges = df_storage_edges[df_storage_edges['src'].isin(third_party_node_names)]

    return df_storage_edges

def remove_storage_edge(df_chosen_edge, df_graph_vid, G, third_party_node_names, original_tlu, ct, result_dir, visit_id, 
    ldb, feature_config, clf, mapping_dict, filterlists, filterlist_rules):

    """
    Function to return a mutated graph by storage removal.

    Args:
        df_chosen edge: Chosen storage edge
        df_graph_vid: DataFrame representing graph
        G: Graph in networkx format
        third_party_node_names: List of adversary URLs
        original_tlu: Top level URL
        ct: Iteration number
        result_dir: Results folder path
        visit_id: Visit ID
        ldb: Content LDB
        feature_config: Feature config
        clf: Classifier
        mapping_dict: Dictionary of content mutation mappings
        filterlists: Names of filter lists to label nodes
        filterlist_rules: Filter list rules to label nodes.
    Returns:
        df: DataFrame of switches in classification 
        df_new_edges: DataFrame of updated edges
        

    This functions does the following:

    1. Removes storage edge.
    2. Perform classification.
    3. Compare with original classifications (original = content mutated graph)
    4. Return results.
    """

    df_new_edges = pd.concat([df_graph_vid, df_chosen_edge], ignore_index=True).drop_duplicates(keep=False)
    #G_new = build_graph(df_new_edges)
    G_new = G.copy()
    chosen_edge_info = df_chosen_edge.iloc[0]
    try:
        G_new.remove_edge(chosen_edge_info['src'], chosen_edge_info['dst'])
    except Exception as e:
        logger.error("Error removing storage edge: " + str(e))

    folder_tag = str(ct) + "_" + str(random.randint(0,10000)) + "_remove_storage"
    num_test = extract_and_classify(df_new_edges, G_new, result_dir, folder_tag, visit_id, ldb, feature_config, clf, filterlists, filterlist_rules)
    new_dirname = os.path.join(result_dir, folder_tag)
    df_original = read_prediction_df(os.path.join(result_dir, "0_content_mutated", "tp_0"))
    df_new = read_prediction_df(os.path.join(new_dirname, "tp_0"))
    result_dict, diff = calculate_misclassifications_mutated(df_new, df_original, mapping_dict, third_party_node_names, original_tlu)
    result_dict = json.dumps(result_dict)

    G_data = json_graph.node_link_data(G_new)
    G_json = json.dumps(G_data)
    change_info = {}
    change_info['type'] = 'storage_edge_removal'
    change_info['info'] = df_chosen_edge.to_dict('index')

    data = {'result_dict': [result_dict], 
            'diff': [diff], 
            'graph': [G_json], 
            'change_info': [change_info],
            'folder_tag': [folder_tag],
            }
    df = pd.DataFrame.from_dict(data)

    return df, df_new_edges

def find_url_receivers(df_graph_vid, third_party_node_names):

    df_cookie_set = df_graph_vid[(df_graph_vid['action'] == 'set') | (df_graph_vid['action'] == 'set_js')]
    df_cookie_set['cookie_val'] = df_cookie_set['attr'].apply(fs.get_cookieval)
    cookie_values = list(set(df_cookie_set['cookie_val'].tolist()))
    urls_to_obfuscate = {}
    
    for dest in third_party_node_names:
        for cookie_value in cookie_values:
            to_replace = find_cookie_in_string(cookie_value, dest)
            if len(to_replace) > 0:
                if dest not in urls_to_obfuscate:
                    urls_to_obfuscate[dest] = []
                urls_to_obfuscate[dest] += to_replace
                break

    return urls_to_obfuscate


def obfuscate_url(dest, to_replace, df_graph_vid, G, third_party_node_names, original_tlu, ct, result_dir, visit_id, 
    ldb, feature_config, clf, mapping_dict, filterlists, filterlist_rules):

    """
    Function to return a mutated graph by URL obfuscation.

    Args:
        dest: Original URL
        to_replace: New URL
        df_graph_vid: DataFrame representing graph
        G: Graph in networkx format
        third_party_node_names: List of adversary URLs
        original_tlu: Top level URL
        ct: Iteration number
        result_dir: Results folder path
        visit_id: Visit ID
        ldb: Content LDB
        feature_config: Feature config
        clf: Classifier
        mapping_dict: Dictionary of content mutation mappings
        filterlists: Names of filter lists to label nodes
        filterlist_rules: Filter list rules to label nodes.
    Returns:
        df: DataFrame of switches in classification 
        df_new_edges: DataFrame of updated edges
        

    This functions does the following:

    1. Updates URL to obfuscate cookie values.
    2. Perform classification.
    3. Compare with original classifications (original = content mutated graph)
    4. Return results.
    """

    new_url = dest
    for item in to_replace:
        to_change = new_url[item[0]:item[0]+item[1]]
        new_url = new_url[:item[0]] + obfuscate_string(to_change) + new_url[item[0]+item[1]:]

    mapping_dict = {dest : new_url}
    df_graph_new = df_graph_vid.copy().replace({'name' : mapping_dict, 'src' : mapping_dict, 'dst' : mapping_dict})
    G_new = gs.build_networkx_graph(df_graph_new)

    folder_tag = str(ct) + "_" + str(random.randint(0,10000)) + "_obfuscate_url"
    num_test = extract_and_classify(df_graph_new, G_new, result_dir, folder_tag, visit_id, ldb, feature_config, clf, filterlists, filterlist_rules)
    
    new_dirname = os.path.join(result_dir, folder_tag)
    df_original = read_prediction_df(os.path.join(result_dir, "0_content_mutated", "tp_0"))
    df_new = read_prediction_df(os.path.join(new_dirname, "tp_0"))
    result_dict, diff = calculate_misclassifications_mutated(df_new, df_original, mapping_dict, third_party_node_names, original_tlu)
    result_dict = json.dumps(result_dict)

    G_data = json_graph.node_link_data(G_new)
    G_json = json.dumps(G_data)
    change_info = {}
    change_info['type'] = 'obfuscate_url'
    change_info['info'] = mapping_dict

    data = {'result_dict': [result_dict], 
            'diff': [diff], 
            'graph': [G_json], 
            'change_info': [change_info],
            'folder_tag': [folder_tag],
            }
    df = pd.DataFrame.from_dict(data)

    return df, df_graph_new


def find_redirect_edges(df_graph_vid, third_party_node_names):

    http_status = [300, 301, 302, 303, 307, 308]
    http_status += [str(x) for x in http_status]
    df_edge = df_graph_vid[(df_graph_vid["graph_attr"] == "Edge") | (df_graph_vid["graph_attr"] == "EdgeWG")]
    df_redirects = df_edge[df_edge['response_status'].isin(http_status)]
    df_redirects = df_redirects[df_redirects['dst'].isin(third_party_node_names)]

    return df_redirects


def redistribute_redirect_edge(df_chosen_edge, df_graph_vid, G, third_party_node_names, original_tlu, ct, result_dir, visit_id, 
    ldb, feature_config, clf, mapping_dict, filterlists, filterlist_rules):

    """
    Function to return a mutated graph by redirect redistribution.

    Args:
        df_chosen_edge: Edge to be changed.
        df_graph_vid: DataFrame representing graph
        G: Graph in networkx format
        third_party_node_names: List of adversary URLs
        original_tlu: Top level URL
        ct: Iteration number
        result_dir: Results folder path
        visit_id: Visit ID
        ldb: Content LDB
        feature_config: Feature config
        clf: Classifier
        mapping_dict: Dictionary of content mutation mappings
        filterlists: Names of filter lists to label nodes
        filterlist_rules: Filter list rules to label nodes.
    Returns:
        df: DataFrame of switches in classification 
        df_new_edges: DataFrame of updated edges
        

    This functions does the following:

    1. Changes redirect pattern.
    2. Perform classification.
    3. Compare with original classifications (original = content mutated graph)
    4. Return results.
    """

    http_status = [300, 301, 302, 303, 307, 308]
    http_status += [str(x) for x in http_status]
    second_limit = 1

    df_edge = df_graph_vid[(df_graph_vid["graph_attr"] == "Edge") | (df_graph_vid["graph_attr"] == "EdgeWG")]
    df_edge['time_stamp_formatted'] = df_edge['time_stamp'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
    df_chosen_edge['time_stamp_formatted'] = df_chosen_edge['time_stamp'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
    original_src = df_chosen_edge.iloc[0]['src']

    ts_to_check = df_chosen_edge['time_stamp_formatted'].iloc[0]
    nodes_to_check = [df_chosen_edge['dst'].iloc[0]]
    flag = True

    while flag:
        limit = ts_to_check + datetime.timedelta(seconds=second_limit)
        df_sent_redirects = df_edge[(df_edge['src'].isin(nodes_to_check) & \
                                        (df_edge['response_status']).isin(http_status))]
        df_sent_redirects = df_sent_redirects[(df_sent_redirects['time_stamp_formatted'] >= ts_to_check) & \
                                        (df_sent_redirects['time_stamp_formatted'] < limit)]
        if len(df_sent_redirects) == 0:
            flag = False
        else:
            df_chosen_edge = df_chosen_edge.append(df_sent_redirects)
            nodes_to_check = df_sent_redirects['dst'].tolist()
            ts_to_check += datetime.timedelta(seconds=second_limit)

    df_new_edges = pd.concat([df_graph_vid, df_chosen_edge], ignore_index=True).drop_duplicates(keep=False)
    df_new_edges = df_new_edges.drop(columns=['time_stamp_formatted'])
    df_chosen_edge = df_chosen_edge.drop(columns=['time_stamp_formatted']).drop_duplicates()

    G_new = G.copy()
    src_list = df_chosen_edge['src'].tolist()
    dst_list = df_chosen_edge['dst'].tolist()
    for i in range(0, len(src_list)):
        print(src_list[i], dst_list[i])
    logger.info("Redirect removal: removing " + str(df_chosen_edge))
    for i in range(0, len(src_list)):
        try:
            G_new.remove_edge(src_list[i], dst_list[i])
        except Exception as e:
            logger.error("Error removing redirect: " + str(e))

    df_chosen_edge = df_chosen_edge.replace(df_chosen_edge['src'].tolist(), [original_src] * len(df_chosen_edge))
    df_chosen_edge = df_chosen_edge.replace(df_chosen_edge['response_status'].tolist(), [200] * len(df_chosen_edge))

    for index, row in df_chosen_edge.iterrows():
        logger.info("Request addition: adding " + row['src'] + " " + row['dst'])
        try:
            G_new.add_edge(row['src'], row['dst'], visit_id=row['visit_id'], \
                action=row['action'], reqattr=row['reqattr'], \
                respattr=row['respattr'], attr=row['attr'],\
                response_status=row['response_status'], time_stamp=row['time_stamp'])
        except Exception as e:
            logger.error("Error adding request: " + str(e))

    #G_new = build_graph(df_new_edges)

    df_new_edges = pd.concat([df_graph_vid, df_chosen_edge], ignore_index=True).drop_duplicates()

    folder_tag = str(ct) + "_" + str(random.randint(0,10000)) + "_remove_redirect"
    num_test = extract_and_classify(df_new_edges, G_new, result_dir, folder_tag, visit_id, ldb, feature_config, clf, filterlists, filterlist_rules)
 
    new_dirname = os.path.join(result_dir, folder_tag)
    df_original = read_prediction_df(os.path.join(result_dir, "0_content_mutated", "tp_0"))
    df_new = read_prediction_df(os.path.join(new_dirname, "tp_0"))
    result_dict, diff = calculate_misclassifications_mutated(df_new, df_original, mapping_dict, third_party_node_names, original_tlu)
    result_dict = json.dumps(result_dict)

    G_data = json_graph.node_link_data(G_new)
    G_json = json.dumps(G_data)
    change_info = {}
    change_info['type'] = 'redirect_removal'
    change_info['info'] = df_chosen_edge.to_dict('index')

    data = {'result_dict': [result_dict], 
            'diff': [diff], 
            'graph': [G_json], 
            'change_info': [change_info],
            'folder_tag': [folder_tag],
            }
    df = pd.DataFrame.from_dict(data)

    return df, df_new_edges
    