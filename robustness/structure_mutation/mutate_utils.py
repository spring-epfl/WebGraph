from openwpm_utils import domain as du
from six.moves.urllib.parse import urlparse, parse_qs
import random
import string
import pandas as pd
import json
import os
from collections import Counter 
from datetime import datetime
from random import randint
from yaml import full_load
import re
import base64
import hashlib 

import graph as gs
from features.feature_extraction import extract_graph_features
import labelling as ls
import classification as cs
from obfuscation import *

import logging
from logging.config import fileConfig

fileConfig('logging.conf')
logger = logging.getLogger(__name__)

def load_config_info(filename):
    
    with open(filename) as file:
        return full_load(file)

def read_graph_file(graph_fname, visit_id):

    df_graph = pd.read_csv(graph_fname)
    df_graph = df_graph[df_graph['visit_id'] == visit_id]
    return df_graph

def check_third_party(row):

    try:
        nodetype = row['type']
        if (nodetype == 'Request') or (nodetype == 'Script') or (nodetype == 'Document'):
            base_domain = row['domain']
            top_level_domain = row['top_level_domain']
            if base_domain and top_level_domain:
                if base_domain != top_level_domain:
                    return True
    except Exception as e:
        return False
    return False

def choose_third_party(df_third_party, df_third_party_ads):

    entities = df_third_party_ads[df_third_party_ads['domain'].notnull()]['domain'].tolist()
    if len(entities) > 0:
        choose_third_party = Counter(entities).most_common()[0][0]
        df_chosen_nodes = df_third_party[df_third_party['domain'] == choose_third_party]
        return df_chosen_nodes
    else:
        return None

def find_third_parties_domain(df, pred_dict):

    df_third_party = find_all_third_parties(df)
    df_third_party_ads = pd.DataFrame()
    
    if len(df_third_party) > 0:
        df_third_party['choose'] = df_third_party.apply(get_ads, args=(pred_dict,), axis=1)
        df_third_party_ads = df_third_party[df_third_party['choose'] == True]
        df_third_party_ads = df_third_party_ads.drop(columns=['choose'])
    return df_third_party, df_third_party_ads

def find_all_third_parties(df):

    df_third_party = df[(df['is_third_party'] == True) & (df['top_level_url'].notnull())]
    return df_third_party

def get_tp_nodes(df, pred_dict, choice=True):

    df['is_third_party'] = df.apply(check_third_party, axis=1)
    df_chosen_nodes = pd.DataFrame()

    if choice:
        df_third_party, df_third_party_ads = find_third_parties_domain(df, pred_dict)
        if len(df_third_party_ads) > 0:
            df_chosen_nodes = choose_third_party(df_third_party, df_third_party_ads)
    else:
        df_third_party = find_all_third_parties(df)
        df_chosen_nodes = df_third_party

    return df_chosen_nodes

def read_predictions(pred_file):

    pred_dict = {}
    with open(pred_file) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split("|$|")
            if len(parts) == 4:
                vid = parts[3].strip()
                name = parts[2].strip()
                pred = parts[1].strip()
                key = vid + "_" + name
                if key not in pred_dict:
                    pred_dict[key] = pred
    return pred_dict


def extract_and_classify(df_graph_vid, G, result_dir, folder_tag, visit_id, 
    ldb, feature_config, clf, filterlists, filterlist_rules):

    test_result_dir = os.path.join(result_dir, folder_tag)
    if not os.path.exists(test_result_dir):
        os.mkdir(test_result_dir)
    
    df_features = extract_graph_features(df_graph_vid, G, visit_id, ldb, feature_config)
    logger.info("Extracted features")
    df_labels = ls.label_data(df_graph_vid, filterlists, filterlist_rules)
    logger.info("Labelled data")
    
    df_labelled = df_features.merge(df_labels[['visit_id', 'name', 'label']], on=['visit_id', 'name'])
    df_labelled = df_labelled[df_labelled['label'] != "Error"]
    df_labelled = df_labelled[~df_labelled['name'].str.contains('_fake')]
    feature_list = feature_config['feature_columns'][2:]
    result =  cs.classify_with_model(clf, df_labelled, test_result_dir, feature_list)
    report = cs.describe_classif_reports([result], test_result_dir)
    #print_stats(report, test_result_dir)

    return len(df_labelled)

def read_prediction_df(fname):
    
    data_dict = {}
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line_parts = line.strip().split('|$|')
            true_val = line_parts[0].strip()
            pred_val = line_parts[1].strip()
            name = line_parts[2].strip()
            visit_id = line_parts[3].strip()
            key = visit_id + "_" + name
            data_dict[key] = {'name' : name, 'visit_id' : visit_id, 'truth' : true_val, 'pred' : pred_val}
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df["visit_id"] = pd.to_numeric(df["visit_id"])
    return df

def mutate_content(df_graph_vid, df_nodes, adv_node_names, top_level_url):

    mutated_adv_node_names = []

    mapping_dict = {}
    party_value = 1
    for nn in adv_node_names:
        obfs_name = obfuscate(nn, party_value, top_level_url)
        mapping_dict[nn] = obfs_name
        mutated_adv_node_names.append(obfs_name)

    df_graph_new = df_graph_vid.copy().replace({'name' : mapping_dict, 'src' : mapping_dict, 'dst' : mapping_dict})
    df_nodes_mut = df_nodes.copy().replace({'name' : mapping_dict})
    G_new = gs.build_networkx_graph(df_graph_new)
    
    return G_new, df_graph_new, df_nodes_mut, mapping_dict, mutated_adv_node_names

def id_generator(size=6, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def create_child_choose_type(child_type, original_url, original_vid, tlu):

    cpt = str(random.randint(3, 22))

    node_attr = \
    {
        "Script" : ['{"content_policy_type": 2, "top_level_url": "' + tlu + '"}'],
        "Request" : ['{"content_policy_type": ' + cpt + ', "top_level_url": "' + tlu + '"}'],
        #"Storage" : {'attr' : 'Cookie'}
    }

    parsed_url = urlparse(original_url)
    max_size = 100
    path = id_generator(size=randint(0, max_size))
    max_size = 16
    domain = id_generator(size=randint(0, max_size))

    node_names = \
    {
        "Script" : parsed_url.scheme + "://" + domain + "/" + path + "_fake.js",
        "Request" : parsed_url.scheme + "://" + domain + "/" + path + "_fake.req",
        #"Storage" : "Storage_fake_" + str(random.randint(0, 100000)),
    }

    edge_attr = \
    {
        "Script" : {
                        'visit_id' : original_vid,
                        'action' : 'N/A', 
                        'reqattr' : json.dumps([['Fake-Header', 'fake_request']]), 
                        'respattr' : json.dumps([['Fake-Header', 'fake_response']]),
                        'attr' : json.dumps({'ctype' : 'script', 'clength' : random.randint(0, 1000)}), \
                        'response_status' : 200,
                        'time_stamp' : datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),
                    },
        "Request" : {
                        'visit_id' : original_vid,
                        'action' : 'N/A', 
                        'reqattr' : json.dumps([['Fake-Header', 'fake_request']]), 
                        'respattr' : json.dumps([['Fake-Header', 'fake_response']]),
                        'attr' : json.dumps({'ctype' : 'stylesheet', 'clength' : random.randint(0, 1000)}), \
                        'response_status' : 200,
                        'time_stamp' : datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
                    },
        "Storage" : { 
                        'visit_id' : original_vid,
                        'action' : 'set',  
                        'reqattr' : 'N/A', 
                        'respattr' : 'N/A', \
                        'attr' : json.dumps({'value' : str(random.randint(0, 1000))}), \
                        'response_status' : 'N/A',
                        'time_stamp' : datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
                    }
    }

    child_df = pd.DataFrame()
    edge_dict = {}

    child_visit_id = original_vid
    child_attr = node_attr[child_type]
    child_name = node_names[child_type]

    child_data = (child_name, {'type' : child_type, 'visit_id' : child_visit_id, 'attr' : child_attr})
    child_df = pd.DataFrame({'name' : child_name, 'visit_id' : child_visit_id, 'type' : child_type, 'attr' : json.dumps(child_attr)}, index=[0])

    edge_dict = edge_attr[child_type]
    
    return child_df, edge_dict

def create_child(parent_type, original_url, original_vid, tlu):

    node_type_transitions = \
    {
        "Document" : ["Script", "Request"],
        "Script" : ["Script", "Request"],
        "Request" : ["Script", "Request"],
        "Storage" : []
    }

    cpt = str(random.randint(3, 22))

    node_attr = \
    {
        "Script" : ['{"content_policy_type": 2, "top_level_url": "' + tlu + '"}'],
        "Request" : ['{"content_policy_type": ' + cpt + ', "top_level_url": "' + tlu + '"}'],
        "Storage" : {'attr' : 'Cookie'}
    }

    parsed_url = urlparse(original_url)
    max_size = 100
    path = id_generator(size=randint(0, max_size))
    max_size = 16
    domain = id_generator(size=randint(0, max_size))

    node_names = \
    {
        "Script" : parsed_url.scheme + "://" + domain + "/" + path + "_fake.js",
        "Request" : parsed_url.scheme + "://" + domain + "/" + path + "_fake.req",
        "Storage" : "Storage_fake_" + str(random.randint(0, 100000)),
    }

    edge_attr = \
    {
        "Script" : {
                        'visit_id' : original_vid,
                        'action' : 'N/A', 
                        'reqattr' : json.dumps([['Fake-Header', 'fake_request']]), 
                        'respattr' : json.dumps([['Fake-Header', 'fake_response']]),
                        'attr' : json.dumps({'ctype' : 'script', 'clength' : random.randint(0, 1000)}), \
                        'response_status' : 200,
                        'time_stamp' : datetime.today().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        'graph_attr' : 'Edge'
                    },
        "Request" : {
                        'visit_id' : original_vid,
                        'action' : 'N/A', 
                        'reqattr' : json.dumps([['Fake-Header', 'fake_request']]), 
                        'respattr' : json.dumps([['Fake-Header', 'fake_response']]),
                        'attr' : json.dumps({'ctype' : cpt, 'clength' : random.randint(0, 1000)}), \
                        'response_status' : 200,
                        'time_stamp' : datetime.today().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        'graph_attr' : 'Edge'
                    },
        "Storage" : { 
                        'visit_id' : original_vid,
                        'action' : 'set',  
                        'reqattr' : 'N/A', 
                        'respattr' : 'N/A', \
                        'attr' : json.dumps({'value' : str(random.randint(0, 1000))}), \
                        'response_status' : 'N/A',
                        'time_stamp' : datetime.today().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                    }
    }

    child_df = pd.DataFrame()
    edge_dict = {}

    if parent_type != "Storage":
        node_type = parent_type
        child_type = random.choice(node_type_transitions[node_type])
        child_visit_id = original_vid
        child_attr = node_attr[child_type]
        child_name = node_names[child_type]

        child_data = (child_name, {'type' : child_type, 'visit_id' : child_visit_id, 'attr' : child_attr})
        child_df = pd.DataFrame({'name' : child_name, 'visit_id' : child_visit_id, \
            'type' : child_type, 'attr' : json.dumps(child_attr), \
            'graph_attr' : 'Node'}, index=[0])

        edge_dict = edge_attr[child_type]
    
    return child_df, edge_dict


def find_full_cookie(cookie_value, url):
    if (len(cookie_value) > 3) and (cookie_value) in url:
        return (url.find(cookie_value), len(cookie_value))
    return (-1, -1)

def find_partial_cookie(cookie_value, url):
    
    split_cookie = re.split(r'\.+|;+|]+|\!+|\@+|\#+|\$+|\%+|\^+|\&+|\*+|\(+|\)+|\-+|\_+|\++|\~+|\`+|\@+=|\{+|\}+|\[+|\]+|\\+|\|+|\:+|\"+|\'+|\<+|\>+|\,+|\?+|\/+', cookie_value)
    for item in split_cookie:
        if (len(item) > 3) and (item in url):
            return (url.find(item), len(item))
    return (-1, -1)
                
def find_md5_cookie(cookie_value, url):
    
    md5_value = hashlib.md5(cookie_value.encode('utf-8')).hexdigest()
    if (len(cookie_value) > 3) and (md5_value in url):
        return (url.find(md5_value), len(md5_value))
    return (-1, -1)

def find_sha1_cookie(cookie_value, url):
    
    sha1_value = hashlib.sha1(cookie_value.encode('utf-8')).hexdigest()
    if (len(cookie_value) > 3) and (sha1_value in url):
        return (url.find(md5_value), len(md5_value))
    return (-1, -1)
            
def find_base64_cookie(cookie_value, url):
    
    base64_value = base64.b64encode(cookie_value.encode('utf-8')).decode('utf8')
    if (len(cookie_value) > 3) and (base64_value in url):
        return (url.find(base64_value), len(base64_value))
    return (-1, -1)

def find_cookie_in_string(cookie_value, dest):

    to_replace = []

    full_cookie = find_full_cookie(cookie_value, dest)
    if full_cookie[0] != -1:
        to_replace.append(full_cookie)
    else:
        partial_cookie = find_partial_cookie(cookie_value, dest)
        if partial_cookie[0] != -1:
            to_replace.append(partial_cookie)
    base64_cookie = find_base64_cookie(cookie_value, dest)
    if base64_cookie[0] != -1:
        to_replace.append(base64_cookie)
    md5_cookie = find_md5_cookie(cookie_value, dest)
    if md5_cookie[0] != -1:
        to_replace.append(md5_cookie)
    sha1_cookie = find_sha1_cookie(cookie_value, dest)
    if sha1_cookie[0] != -1:
        to_replace.append(sha1_cookie)

    return to_replace

def find_third_parties(df):

    try:
        base_domain = du.get_ps_plus_1(df['name'])
        top_level_domain = du.get_ps_plus_1(df['top_level_url'])

        if df['name'] and df['top_level_url']:
            if base_domain != top_level_domain:
                return True
            else:
                return False
        else:
            return "unknown"
    except Exception as e:
        print(e)

def is_third_party(domain, tlu):

    try:
        tlu_domain = du.get_ps_plus_1(tlu)
        if domain and tlu:
            if domain != tlu:
                return True
            else:
                return False
        else:
            return "unknown"
    except:
        return "unknown"

def find_domain(name):
    try:
        return du.get_ps_plus_1(name)
    except:
        return None

def get_ads(df, pred_dict):

    key = str(int(df['visit_id'])) + "_" + df['name']
    if key in pred_dict:
        pred = pred_dict[key]
        if pred == "True":
            return True
        else:
            return False
    else:
        return None

def get_mapping(name, mapping_dict):

    try:
        return mapping_dict[name]
    except:
        return name

def write_results_to_diff_file(ct, diff_file, result_dict):

    with open(diff_file, "a") as f:
        f.write(
            "Iteration: " + str(ct) + \
            " Diff: " + str(result_dict['diff']) + \
            " Desired: " + str(result_dict['total_desired']) + 
            " Undesired: " + str(result_dict['total_undesired']) + \
            " ad_adv_content_adv: " + str(result_dict['ad_adv_content_adv']) + \
            " ad_others_content_others: " + str(result_dict['ad_others_content_others']) + \
            " ad_tp_content_tp: " + str(result_dict['ad_tp_content_tp']) + \
            " ad_fp_content_fp: " + str(result_dict['ad_fp_content_fp']) + \
            " content_others_ad_others: " + str(result_dict['content_others_ad_others']) + \
            " content_tp_ad_tp: " + str(result_dict['content_tp_ad_tp']) + \
            " content_fp_ad_fp: " + str(result_dict['content_fp_ad_fp']) + \
            " content_adv_ad_adv: " + str(result_dict['content_adv_ad_adv']) + \
            "\n")

def write_successful_info_to_file(ct, success_dir, df_chosen, convert=True):

    success_file = os.path.join(success_dir, "graph_" + str(ct) + ".json")

    with open(success_file, "w") as f:
        f.write(df_chosen['graph'])

    success_file = os.path.join(success_dir, "change_" + str(ct) + ".json")

    with open(success_file, "w") as f:
        f.write(json.dumps(df_chosen['change_info']))

def calculate_misclassifications_mutated(df, df_original, mapping_dict, adv_node_names, tlu):

    #Get mappings of mutated URLs
    
    #Divide into adversary and non-adversary nodes
    df_original_adv = df_original[df_original['name'].isin(adv_node_names)]
    df_new_adv = df[df['name'].isin(adv_node_names)]
    df_adv = pd.merge(df_original_adv, df_new_adv, how='inner', left_on=['visit_id', 'name'], right_on=['visit_id', 'name'])
    
    df_original_others = df_original[~df_original['name'].isin(adv_node_names)]
    df_new_others = df[~df['name'].isin(adv_node_names)]
    df_others = pd.merge(df_original_others, df_new_others, how='inner', left_on=['visit_id', 'name'], right_on=['visit_id', 'name'])
    
    #Mark non-adversary nodes as first or third party
    df_others['domain'] = df_others['name'].apply(find_domain)
    df_others['is_third_party'] = df_others['domain'].apply(is_third_party, tlu=tlu)

    result_dict = {}

    #Find desired switches 
    result_dict['ad_adv_content_adv'] = len(df_adv[(df_adv['pred_x'] == 'True') & (df_adv['pred_y'] == 'False')])
    
    #Find undesired switches
    result_dict['ad_others_content_others'] = len(df_others[(df_others['pred_x'] == 'True') & (df_others['pred_y'] == 'False')])
    result_dict['ad_tp_content_tp'] = len(df_others[(df_others['pred_x'] == 'True') & (df_others['pred_y'] == 'False') & (df_others['is_third_party'] == True)])
    result_dict['ad_fp_content_fp'] = len(df_others[(df_others['pred_x'] == 'True') & (df_others['pred_y'] == 'False') & (df_others['is_third_party'] == False)])
    
    result_dict['content_others_ad_others'] = len(df_others[(df_others['pred_x'] == 'False') & (df_others['pred_y'] == 'True')])
    result_dict['content_tp_ad_tp'] = len(df_others[(df_others['pred_x'] == 'False') & (df_others['pred_y'] == 'True') & (df_others['is_third_party'] == True)])
    result_dict['content_fp_ad_fp'] = len(df_others[(df_others['pred_x'] == 'False') & (df_others['pred_y'] == 'True') & (df_others['is_third_party'] == False)])
    
    result_dict['content_adv_ad_adv'] = len(df_adv[(df_adv['pred_x'] == 'False') & (df_adv['pred_y'] == 'True')])
    
    result_dict['total_desired'] = result_dict['ad_adv_content_adv']
    result_dict['total_undesired'] = result_dict['content_others_ad_others'] + result_dict['content_adv_ad_adv']
    # Alternate for the experiment: adversary does not care about other nodes
    #result_dict['total_undesired'] = result_dict['content_adv_ad_adv']

    result_dict['diff'] = result_dict['total_desired'] - result_dict['total_undesired']
    
    return result_dict, result_dict['diff']

def calculate_misclassifications(df, df_original, third_party_node_names):

    df_original_third_party = df_original[df_original['name'].isin(third_party_node_names)]
    df_new_third_party = df[df['name'].isin(third_party_node_names)]
    df_third_party = pd.merge(df_original_third_party, df_new_third_party, how='inner', on=['visit_id', 'name'])
    
    df_original_others = df_original[~df_original['name'].isin(third_party_node_names)]
    df_new_others = df[~df['name'].isin(third_party_node_names)]
    df_others = pd.merge(df_original_others, df_new_others, how='inner', on=['visit_id', 'name'])
    
    ad_tp_content_tp = len(df_third_party[(df_third_party['pred_x'] == 'True') & (df_third_party['pred_y'] == 'False')])
    
    ad_others_content_others = len(df_others[(df_others['pred_x'] == 'True') & (df_others['pred_y'] == 'False')])
    content_others_ad_others = len(df_others[(df_others['pred_x'] == 'False') & (df_others['pred_y'] == 'True')])
    content_tp_ad_tp = len(df_third_party[(df_third_party['pred_x'] == 'False') & (df_third_party['pred_y'] == 'True')])
    total_desired = ad_tp_content_tp
    total_undesired = ad_others_content_others + content_others_ad_others + content_tp_ad_tp

    diff = total_desired - total_undesired
    
    return diff, total_desired, total_undesired, ad_tp_content_tp, content_tp_ad_tp, ad_others_content_others, content_others_ad_others


