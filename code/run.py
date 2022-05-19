import argparse
import sys
import traceback
from pathlib import Path
from typing import List
import time

import pandas as pd
from tqdm import tqdm
from yaml import full_load
import leveldb

import graph as gs
from graph.database import Database
import labelling as ls
from feature_extraction import extract_graph_features


pd.set_option("display.max_rows", None, "display.max_columns", None)


def load_config_info(filename):
    """Load features from features.yaml file
    :param filename: yaml file name containing feature names
    :return: list of features to use.
    """
    with open(filename) as file:
        return full_load(file)


def extract_features(pdf, networkx_graph, visit_id, config_info, ldb_file):
    """Getter to generate the features of each node in a graph.
    :param pdf: pandas df of nodes and edges in a graph.
    :param G: Graph object representation of the pdf.
    :return: dataframe of features per node in the graph
    """
    # Generate features for each node in our graph
    ldb = leveldb.LevelDB(ldb_file)
    df_features = extract_graph_features(pdf, networkx_graph, visit_id, ldb, config_info)
    return df_features


def find_setter_domain(setter):
    try:
        domain = gs.get_domain(setter)
        return domain
    except:
        return None


def find_domain(row):

    domain = None

    try:
        node_type = row['type']
        if (node_type == 'Document') or (node_type == 'Request') or (node_type == 'Script'):
            domain = gs.get_domain(row['name'])
        elif (node_type == 'Element'):
            return domain
        else:
            return row['domain']
        return domain
    except Exception as e:
        traceback.print_exc()
        return None


def find_tld(top_level_url):
    try:
        if top_level_url:
            tld = gs.get_domain(top_level_url)
            return tld
        else:
            return None
    except:
        return None

def get_party(row):

    if row['type'] == 'Storage':
        if row['domain'] and row['top_level_domain']:
            if row['domain'] == row['top_level_domain']:
                return 'first'
            else:
                return 'third'
    return "N/A"

def find_setters(df_all_storage_nodes, df_http_cookie_nodes, df_all_storage_edges, df_http_cookie_edges):

    df_setter_nodes = pd.DataFrame(columns=['visit_id', 'name', 'type', 'attr', 'top_level_url', 'domain', 'setter', 'setting_time_stamp'])

    try:

        df_storage_edges = pd.concat([df_all_storage_edges, df_http_cookie_edges])
        if len(df_storage_edges) > 0:
            df_storage_sets = df_storage_edges[(df_storage_edges['action'] == 'set') 
                                | (df_storage_edges['action'] == 'set_js')]
            df_setters = gs.get_original_cookie_setters(df_storage_sets)
            df_storage_nodes = pd.concat([df_all_storage_nodes, df_http_cookie_nodes])
            df_setter_nodes = df_storage_nodes.merge(df_setters, on=['visit_id', 'name'], how='outer')
        
    except Exception as e:
        print("Error getting setter:", e)
        traceback.print_exc()

    return df_setter_nodes


def build_graph(database: Database, visit_id):
    """Read SQL data from crawler for a given visit_ID.
    :param visit_id: visit ID of a crawl URL.
    :return: Parsed information (nodes and edges) in pandas df.
    """
    # Read tables from DB and store as DataFrames
    df_requests, df_responses, df_redirects, call_stacks, javascript = database.website_from_visit_id(visit_id)
    df_js_nodes, df_js_edges = gs.build_html_components(javascript)
    df_request_nodes, df_request_edges = gs.build_request_components(df_requests, df_responses, df_redirects, call_stacks)
    df_all_storage_nodes, df_all_storage_edges = gs.build_storage_components(javascript)
    df_http_cookie_nodes, df_http_cookie_edges = gs.build_http_cookie_components(df_request_edges, df_request_nodes)
    df_storage_node_setter = find_setters(df_all_storage_nodes, df_http_cookie_nodes, df_all_storage_edges, df_http_cookie_edges)

    # Concatenate to get all nodes and edges
    df_request_nodes['domain'] = None
    df_all_nodes = pd.concat([df_js_nodes, df_request_nodes, df_storage_node_setter])
    df_all_nodes['domain'] = df_all_nodes.apply(find_domain, axis=1)
    df_all_nodes['top_level_domain'] = df_all_nodes['top_level_url'].apply(find_tld)
    df_all_nodes['setter_domain'] = df_all_nodes['setter'].apply(find_setter_domain)
    df_all_nodes = df_all_nodes.drop_duplicates()
    df_all_nodes['graph_attr'] = "Node"
    
    df_all_edges = pd.concat([df_js_edges, df_request_edges, df_all_storage_edges, df_http_cookie_edges])
    df_all_edges = df_all_edges.drop_duplicates()
    df_all_edges['top_level_domain'] = df_all_edges['top_level_url'].apply(find_tld)
    df_all_edges['graph_attr'] = "Edge"

    #Remove all non-FP cookies, comment for unblocked
    df_all_nodes['party'] = df_all_nodes.apply(get_party, axis=1)
    third_parties = df_all_nodes[df_all_nodes['party'] == 'third']['name'].unique()
    df_all_nodes = df_all_nodes[~df_all_nodes['name'].isin(third_parties)]
    df_all_edges = df_all_edges[~df_all_edges['dst'].isin(third_parties)]
    df_all_edges = df_all_edges[~df_all_edges['src'].isin(third_parties)]

    df_all_graph = pd.concat([df_all_nodes, df_all_edges])
    df_all_graph = df_all_graph.astype(
        {
            'type' : 'category',
            'response_status' : 'category'
        }
    )

    return df_all_graph


def label_data(df, filterlists, filterlist_rules):
    df_labelled = pd.DataFrame()

    try:
        df_nodes = df[df['graph_attr'] == "Node"]
        df_labelled = ls.label_nodes(df_nodes, filterlists, filterlist_rules)
    except Exception as e:
        print("Error labelling:", e)

    return df_labelled

def apply_tasks(df, visit_id, config_info, ldb_file, output_dir, overwrite):

    # Build the graph
    print(df.iloc[0]['top_level_url'], visit_id, len(df))
    graph_columns = config_info['graph_columns']
    feature_columns = config_info['feature_columns']
    
    try:
        start = time.time()
        graph_path = output_dir / "graph.csv"
        if overwrite or not graph_path.is_file():
            df.reindex(columns=graph_columns).to_csv(str(graph_path))
        else:
            df.reindex(columns=graph_columns).to_csv(str(graph_path), mode='a', header=False)
        
        networkx_graph = gs.build_networkx_graph(df)
        
        df_features = extract_features(df, networkx_graph, visit_id, config_info, ldb_file)
        features_path = output_dir / "features.csv"
        
        if overwrite or not features_path.is_file():
            df_features.reindex(columns=feature_columns).to_csv(str(features_path))
        else:
            df_features.reindex(columns=feature_columns).to_csv(str(features_path), mode='a', header=False)
        end = time.time()
        print("Extracted features:", end-start)

        # #Label data
        # df_labelled = label_data(pdf, filterlists, filterlist_rules)
        # if len(df_labelled) > 0:
        #     df_labelled_path = output_dir / "labelled.csv"
        #     if overwrite or not df_labelled_path.is_file():
        #         df_labelled.to_csv(str(df_labelled_path))
        #     else:
        #         df_labelled.to_csv(str(df_labelled_path), mode='a', header=False)
        
    except Exception as e:
        print("Errored in pipeline:", e)
        traceback.print_exc()


def pipeline(db_file: Path, ldb_file, features_file, filterlist_dir: Path, output_dir: Path, overwrite=True):
    
    number_failures = 0

    #ls.download_lists(filterlist_dir, overwrite)
    #filterlists, filterlist_rules = ls.create_filterlist_rules(filterlist_dir)
    config_info = load_config_info(features_file)

    output_dir.mkdir(parents=True, exist_ok=True)

    with Database(db_file) as database:
        try:
            sites_visits = database.sites_visits()
        except Exception as e:
            tqdm.write(f"Problem reading the sites_visits: {e}")
            exit()


        for _, row in tqdm(sites_visits.iterrows(), total=len(sites_visits), position=0, leave=True, ascii=True):
            # For each visit, grab the visit_id and the site_url
            visit_id = row['visit_id']
            tqdm.write("")
            tqdm.write(f"• Visit ID: {visit_id}")

            try:
                start = time.time()
                pdf = build_graph(database, visit_id)
                tqdm.write(str(pdf.shape))
                pdf.groupby(['visit_id', 'top_level_domain']).apply(apply_tasks, visit_id, config_info, ldb_file, output_dir, overwrite)
                end = time.time()
                print("Done!", end - start)

            except Exception as e:
                number_failures += 1
                tqdm.write(f"Fail: {number_failures}")
                tqdm.write(f"Error: {e}")
                traceback.print_exc()

            #break

    percent = (number_failures/len(sites_visits))*10
    print(f"Fail: {number_failures}, Total: {len(sites_visits)}, Percentage:{percent}", db_file)


def main(program: str, args: List[str]):
    
    parser = argparse.ArgumentParser(prog=program, description="Run a classification pipeline.")
    parser.add_argument(
        "--input-db",
        type=Path,
        help="Input SQLite database.",
        default=Path("crawl-data.sqlite")
    )
    parser.add_argument(
        "--ldb",
        type=str,
        help="Input LDB.",
        default="content.ldb"
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Features.",
        default=Path("features.yaml")
    )
    parser.add_argument(
        "--filters",
        type=Path,
        help="Filters directory.",
        default=Path("filterlists")
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Directory to output the results.",
        default=Path("out")
    )

    ns= parser.parse_args(args)

    pipeline(ns.input_db, ns.ldb, ns.features, ns.filters, ns.out, overwrite=False)


if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])
