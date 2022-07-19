
from obfuscation import *
from mutate_styles import *
from typing import List
from networkx.readwrite import json_graph
import pandas as pd
from yaml import load, dump
import os
import time
import sys
import joblib
import json
import shutil
import leveldb
import argparse
from pathlib import Path
import logging
from logging.config import fileConfig
import traceback

from context import *

fileConfig('logging.conf')
logger = logging.getLogger('root')

def mutate_graph(df_graph_vid, G, df_nodes, third_party_node_names, original_tlu, ct, result_dir, visit_id, \
        ldb, feature_config, clf, mapping_dict, num_parent_nodes, mutation_style, filterlists, filterlist_rules):
    
    """
    Function to perform structural/flow mutations.

    Args:
        df_graph_vid: DataFrame representing graph
        G: Graph in networkx format
        df_nodes: Adversary nodes
        third_party_node_names: List of adversary URLs
        original_tlu: Top level URL
        ct: Iteration number
        result_dir: Results folder path
        visit_id: Visit ID
        ldb: Content LDB
        feature_config: Feature config
        clf: Trained classifier
        mapping_dict: Dictionary of content mutation mappings
        num_parent_nodes: Number of start nodes to run the mutation
        mutation_style: The types of mutations to be run
        filterlists: Names of filter lists to label nodes
        filterlist_rules: Filter list rules to label nodes.

    Returns:
        G_mutate: Mutated networkX graph
        df_graph_mut_new: DataFrame of mutated graph
        df_new_node: Chosen parent starter node
        chosen_type: Most successful chosen mutation type

    This functions does the following:

    1. Create mutated graphs by calling expand_graph on each adversary node.
    2. Calculate the diff value (desired/undesired tradeoff) for each graph.
    3. Select the graph with the maximum diff value.
    4. Write results to files.
    """

    df_all_results = pd.DataFrame()
    df_new_node = pd.DataFrame()
    df_graph_mut_new = pd.DataFrame()
    folder_dict = {}
    node_dict = {}

    if 'node_addition' in mutation_style:
        logger.info('Trying node addition')
        logger.info('Parent nodes: ' + str(num_parent_nodes))
        for i in range(0, num_parent_nodes):
            df_chosen_node = df_nodes.sample().copy()
            df_result, df_graph_vid_new, df_new_node = add_node(df_chosen_node, df_graph_vid, G, third_party_node_names, original_tlu,
                        ct, result_dir, visit_id, ldb, feature_config, clf, mapping_dict, filterlists, filterlist_rules)
            df_all_results = df_all_results.append(df_result)
            folder_dict[df_result['folder_tag'].iloc[0]] = df_graph_vid_new.copy()
            node_dict[df_result['folder_tag'].iloc[0]] = df_new_node
        
    if 'storage_removal' in mutation_style:
        logger.info('Trying storage_removal')
        df_storage_edges = find_storage_edges(df_graph_vid, third_party_node_names)
        logger.info('Number of storage edges found: ' + str(len(df_storage_edges)))
        num_select_edges = min(len(df_storage_edges), num_parent_nodes)
        logger.info('Number of storage edges sampled: ' + str(num_select_edges))
        for i in range(0, num_select_edges):
            df_chosen_edge = df_storage_edges.sample().copy()
            df_result, df_graph_vid_new = remove_storage_edge(df_chosen_edge, df_graph_vid, G, third_party_node_names, original_tlu, 
                ct, result_dir, visit_id, ldb, feature_config, clf, mapping_dict, filterlists, filterlist_rules)
            df_all_results = df_all_results.append(df_result)
            folder_dict[df_result['folder_tag'].iloc[0]] = df_graph_vid_new.copy()

    if 'redirect_removal' in mutation_style:
        logger.info('Trying redirect_removal')
        df_redirect_edges = find_redirect_edges(df_graph_vid, third_party_node_names)
        logger.info('Number of redirect edges found: ' + str(len(df_redirect_edges)))
        num_select_edges = min(len(df_redirect_edges), num_parent_nodes)
        logger.info('Number of redirect edges sampled: ' + str(num_select_edges))   
        for i in range(0, num_select_edges):
            df_chosen_edge = df_redirect_edges.sample().copy()
            df_result, df_graph_vid_new = redistribute_redirect_edge(df_chosen_edge, df_graph_vid, G, third_party_node_names, original_tlu, 
                ct, result_dir, visit_id, ldb, feature_config, clf, mapping_dict, filterlists, filterlist_rules)
            df_all_results = df_all_results.append(df_result)
            folder_dict[df_result['folder_tag'].iloc[0]] = df_graph_vid_new.copy()

    if 'url_obfuscation' in mutation_style:
        logger.info('Trying url_obfuscation')
        urls_to_obfuscate = find_url_receivers(df_graph_vid, third_party_node_names)
        logger.info('Number of URL edges found: ' + str(len(urls_to_obfuscate)))
        num_select_edges = min(len(urls_to_obfuscate), num_parent_nodes)
        logger.info('Number of URL edges sampled: ' + str(num_select_edges))
        keys = list(urls_to_obfuscate.keys())
        for i in range(0, num_select_edges):
            chosen_url = keys.pop()
            df_result, df_graph_vid_new = obfuscate_url(chosen_url, urls_to_obfuscate[chosen_url], df_graph_vid, G, third_party_node_names, original_tlu, 
                ct, result_dir, visit_id, ldb, feature_config, clf, mapping_dict, filterlists, filterlist_rules)
            df_all_results = df_all_results.append(df_result)
            folder_dict[df_result['folder_tag'].iloc[0]] = df_graph_vid_new.copy()

    if len(df_all_results) == 0:
        logger.error('Empty results from mutation')
        return "Empty", "Empty", "Empty", "Empty" 
    
    df_chosen = df_all_results[df_all_results['diff']==df_all_results['diff'].max()].sample().iloc[0]
    
    G_data = json.loads(df_chosen['graph'])
    G_mutate = json_graph.node_link_graph(G_data) 
    
    #Write results to output
    diff_file = os.path.join(result_dir, "diff_stats")
    result_dict = json.loads(df_chosen['result_dict'])
    success_dir = os.path.join(result_dir, "success_info")
    if not os.path.exists(success_dir):
        os.mkdir(success_dir)
    write_results_to_diff_file(ct, diff_file, result_dict)
    write_successful_info_to_file(ct, success_dir, df_chosen, convert=False)
    
    #Remove unsuccessful folders
    folder_list = df_all_results[df_all_results['folder_tag'].notnull()]['folder_tag'].tolist()
    chosen_folder = df_chosen['folder_tag']
    
    for folder_name in folder_list:
        full_path = os.path.join(result_dir, folder_name)
        try:
            if folder_name != chosen_folder:
                #logger.info("remove: " + folder_name)
                shutil.rmtree(full_path)
            else:
                logger.info("Keeping: " + folder_name)
        except OSError as e:
            logger.error("Error in folder removal: " + str(e))

    df_graph_mut_new = folder_dict[chosen_folder]
    chosen_type = df_chosen['change_info']['type']
    if chosen_type == 'node_addition':
        df_new_node = node_dict[chosen_folder]

    return G_mutate, df_graph_mut_new, df_new_node, chosen_type


def pipeline(config):
    """
    Function to run the pipeline for structure mutations. 

    Args:
        filepath_config: Path to config file.
    Returns:
        None

    The pipeline consists of the following steps for each site that we want to mutate.

    1. Build the original graph and get the original predictions of WebGraph for it
       using a trained model.
    2. Get the adversary's nodes on the graph. Default adversary is a third party with
       the largest number of nodes present. This can be changed to all third parties.
       Sample nodes to appropriate value.
    3. Perform content mutation on the graph. 
    4. Run WebGraph, get classification switches. Calculate required number of switches
       for structural mutation.
    5. Run structural/flow mutation for specified number of iterations (here up to 20% growth).
       After each iteration, write results to file and perform re-sampling of parent nodes.

    """

    LDB_PATH = config['content_ldb']
    MODEL_FILE = config['model']
    RESULT_DIR = config['result_dir']
    GRAPH_DATA_FILE = config['graph_data']
    VID_FILE = config['vid_file']
    MUTATION_STYLE = config['mutation_style']
    FEATURE_FILE = config['feature_config']
    FILTERLIST_DIR = config['filterlists']
    PARENT_LIMIT = config['parent_limit']

    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)
    
    feature_config = load_config_info(FEATURE_FILE)
    ldb = leveldb.LevelDB(LDB_PATH)

    with open(VID_FILE) as f:
        vid_list = json.loads(f.read())['vids']
        vid_list = sorted(vid_list)

    lfs.download_lists(Path(FILTERLIST_DIR), overwrite=False)
    filterlists, filterlist_rules = lfs.create_filterlist_rules(Path(FILTERLIST_DIR))

    for visit_id in vid_list:

        try:
            logger.info("START visit_id: " + str(visit_id))

            #Create a new directory for each visit ID
            RESULT_DIR_VID = os.path.join(RESULT_DIR, "vid_" + str(visit_id))
            if not os.path.exists(RESULT_DIR_VID):
                os.mkdir(RESULT_DIR_VID)

            #Load trained WebGraph model
            clf = joblib.load(MODEL_FILE)

            #Read graph for visit_id
            df_graph = read_graph_file(GRAPH_DATA_FILE, visit_id)
            df_graph_vid = df_graph.copy()

            #Build graph
            G_original = gs.build_networkx_graph(df_graph_vid)
            G_copy = G_original.copy()
            logger.info("Built original graph")
            #We limit iterations to 20% of graph size
            num_iterations = int(len(G_copy.nodes()) * 0.2) 
                
            beginning = time.time()

            #Get original predictions
            folder_tag = "original"
            num_test = extract_and_classify(df_graph_vid, G_copy, RESULT_DIR_VID, 
                folder_tag, visit_id, ldb, feature_config, clf, filterlists, filterlist_rules)
            logger.info("Got original predictions")

            #Read predictions and get adversary nodes
            PRED_FILE = os.path.join(RESULT_DIR_VID, folder_tag, "tp_0")
            pred_dict = read_predictions(PRED_FILE)
            df_nodes = get_tp_nodes(df_graph_vid, pred_dict)
            logger.info("Got adversary's nodes")

            if len(df_nodes) > 0:

                df_original = read_prediction_df(PRED_FILE)
                df_max_desired = pd.merge(df_original, df_nodes, how='inner', on=['visit_id', 'name'])
                max_desired = len(df_max_desired[df_max_desired['pred'] == 'True'])

                G_current = G_original.copy()
                logger.info("Number of nodes in graph: " + str(len(G_current.nodes())))
                df_nodes_current = df_nodes.copy()
                num_parent_nodes = min(len(df_nodes_current), PARENT_LIMIT)

                with open(os.path.join(RESULT_DIR_VID, "overall_stats"), "w") as f: 
                    f.write("Number of nodes: " + str(len(G_current.nodes())) + "\n")
                    f.write("Number of edges: " + str(len(G_current.edges())) + "\n")
                    f.write("Maximum possible desired: " + str(max_desired) + "\n")
                    f.write("Total URLs: " + str(num_test) + "\n")
                    f.write("Number of parents: " + str(num_parent_nodes) + "\n\n")

                original_tlu = df_nodes.iloc[0]['top_level_url']
                adversary_node_names = df_nodes['name'].tolist()
                ct = 0

                urls_to_obfuscate = find_url_receivers(df_graph_vid, adversary_node_names)
                logger.info("Cookie URLs in original graph: " + str(len(urls_to_obfuscate)))

                #Content mutation
                G_content_mut, df_graph_mut, df_nodes_mut, mapping_dict, mutated_adv_node_names = \
                    mutate_content(df_graph_vid, df_nodes_current, adversary_node_names, original_tlu)
                logger.info("Performed content mutation on graph")

                urls_to_obfuscate = find_url_receivers(df_graph_mut, mutated_adv_node_names)
                print(len(urls_to_obfuscate), urls_to_obfuscate)

                #Classification and calculation to see how many nodes have to be switched
                cm_folder_tag = str(ct) + "_content_mutated"
                num_test = extract_and_classify(df_graph_mut, G_content_mut.copy(), RESULT_DIR_VID, 
                    cm_folder_tag, visit_id, ldb, feature_config, clf, filterlists, filterlist_rules)
                new_dirname = os.path.join(RESULT_DIR_VID, cm_folder_tag)
                df_original = read_prediction_df(os.path.join(RESULT_DIR_VID, "original", "tp_0"))
                df_new = read_prediction_df(os.path.join(new_dirname, "tp_0"))
                result_dict, diff = calculate_misclassifications_mutated(df_new, df_original, mapping_dict, mutated_adv_node_names, original_tlu)
                diff_file = os.path.join(RESULT_DIR_VID, "diff_stats")
                write_results_to_diff_file(ct, diff_file, result_dict)
                ct += 1

                with open(os.path.join(new_dirname, "mapping_dict.json"), "w") as f:
                    f.write(json.dumps(mapping_dict))

                logger.info("Got results after content mutation")

                G_current = G_content_mut.copy()
                df_nodes_current = df_nodes_mut.copy()
                df_nodes_current_nonstorage = df_nodes_current[df_nodes_current['type'] != "Storage"]
                df_nodes_current_sampled = df_nodes_current_nonstorage.sample(n=num_parent_nodes, random_state=1)

                while ct <= num_iterations:
                    start = time.time()
                    G_mutate, df_graph_mut_new, df_nodes_mutate, chosen_type = mutate_graph(df_graph_mut, G_current, df_nodes_current_nonstorage, \
                        mutated_adv_node_names, original_tlu, ct, RESULT_DIR_VID, visit_id, \
                        ldb, feature_config, clf, mapping_dict, num_parent_nodes, MUTATION_STYLE, filterlists, filterlist_rules)
                    if G_mutate == "Empty":
                        break
                    G_current = G_mutate
                    df_graph_mut = df_graph_mut_new.copy()
                    if len(df_nodes_mutate) > 0:
                        df_nodes_current_nonstorage = df_nodes_current_nonstorage.append(df_nodes_mutate)
                        df_nodes_current_nonstorage = df_nodes_current_nonstorage.drop_duplicates()
                    ct += 1
                    stop = time.time()
                    with open(os.path.join(RESULT_DIR_VID, "overall_stats"), "a") as f: 
                        f.write("Time taken for one iteration: " + str(stop - start) + " sec\n")
                        f.write("Chosen mutation: " + chosen_type + "\n")
                        f.write("Number of nodes in graph: " + str(len(G_current.nodes())) + "\n")
                        f.write("Number of edges in graph: " + str(len(G_current.edges())) + "\n")
            
                end = time.time()
                logger.info("Total time taken: " + str(end - beginning) + " sec")

            else:
                logger.info("No switches required for visit_id: " + str(visit_id))

            logger.info("END visit_id: " + str(visit_id))

        except Exception as e:
            logger.error(e)
            traceback.print_exc()


def main(program: str, args: List[str]):
    
    parser = argparse.ArgumentParser(prog=program, description="Run the structural/flow mutation pipeline.")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Mutation config file.",
        default="config.yaml"
    )

    ns = parser.parse_args(args)
    config = load_config_info(ns.config)
    pipeline(config)


if __name__ == "__main__":

    main(sys.argv[0], sys.argv[1:])
    


