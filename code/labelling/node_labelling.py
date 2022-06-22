from pathlib import Path
import os
import json
import pandas as pd

import requests
from adblockparser import AdblockRules


ADBLOCK_LISTS = {
    'easylist': 'https://easylist.to/easylist/easylist.txt',
    'easyprivacy': 'https://easylist.to/easylist/easyprivacy.txt',
    'antiadblock': 'https://raw.github.com/reek/anti-adblock-killer/master/anti-adblock-killer-filters.txt',
    'blockzilla': 'https://raw.githubusercontent.com/annon79/Blockzilla/master/Blockzilla.txt',
    'fanboyannoyance': 'https://easylist.to/easylist/fanboy-annoyance.txt',
    'fanboysocial': 'https://easylist.to/easylist/fanboy-social.txt',
    'peterlowe': 'http://pgl.yoyo.org/adservers/serverlist.php?hostformat=adblockplus&mimetype=plaintext',
    'squid': 'http://www.squidblacklist.org/downloads/sbl-adblock.acl',
    'warning': 'https://easylist-downloads.adblockplus.org/antiadblockfilters.txt',
}


def read_file_newline_stripped(fname):

    """
    Function to read filter list file.
    
    Args:
        fname: File path.
    Returns:
        lines: List of file lines.
    """

    with open(fname) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def get_resource_type(attr):

    """
    Function to get resource type of a node.
    
    Args:
        attr: Node attributes.
    Returns:
        Resource type of node.
    """

    try:
        attr = json.loads(attr)
        return attr['content_policy_type']
    except Exception as e:
        print("error in type", e)
        return None


def download_lists(filterlist_dir: Path, overwrite: bool = True):
    
    """
    Function to download the filter lists used for labelling.
    
    Args:
        filterlist_dir: Path of the output directory to which filter lists should be written.
    Returns:
        Nothing, writes the lists to a directory.
    
    This functions does the following:
    1. Sends HTTP requests for the filter lists.
    2. Writes to an output directory.
    """

    filterlist_dir.mkdir(exist_ok=True)

    for listname, url in ADBLOCK_LISTS.items():
        list_filename = filterlist_dir / (listname + ".txt")
        if overwrite or not list_filename.exists():
            content = requests.get(url).content
            list_filename.write_bytes(content)


def create_filterlist_rules(filterlist_dir):

    """
    Function to create AdBlockRules objects for the filterlists. 
    
    Args:
        filterlist_dir: Path of the output directory to which filter lists should be written.
    Returns:
        filterlists: List of filter list names.
        filterlist_rules: Dictionary of filter lists and their rules.
    """

    filterlist_rules = {}
    filterlists = os.listdir(filterlist_dir)

    for fname in filterlists:
        rule_dict = {}
        rules = read_file_newline_stripped(os.path.join(filterlist_dir, fname))
        rule_dict['script'] = AdblockRules(rules, use_re2=False, max_mem=1024*1024*1024, supported_options=['script', 'domain', 'subdocument'], skip_unsupported_rules=False)
        rule_dict['script_third'] = AdblockRules(rules, use_re2=False, max_mem=1024*1024*1024, supported_options=['third-party', 'script', 'domain', 'subdocument'], skip_unsupported_rules=False)
        rule_dict['image'] = AdblockRules(rules, use_re2=False, max_mem=1024*1024*1024, supported_options=['image', 'domain', 'subdocument'], skip_unsupported_rules=False)
        rule_dict['image_third'] = AdblockRules(rules, use_re2=False, max_mem=1024*1024*1024, supported_options=['third-party', 'image', 'domain', 'subdocument'], skip_unsupported_rules=False)
        rule_dict['css'] = AdblockRules(rules, use_re2=False, max_mem=1024*1024*1024, supported_options=['stylesheet', 'domain', 'subdocument'], skip_unsupported_rules=False)
        rule_dict['css_third'] = AdblockRules(rules, use_re2=False, max_mem=1024*1024*1024, supported_options=['third-party', 'stylesheet', 'domain', 'subdocument'], skip_unsupported_rules=False)
        rule_dict['xmlhttp'] = AdblockRules(rules, use_re2=False, max_mem=1024*1024*1024, supported_options=['xmlhttprequest', 'domain', 'subdocument'], skip_unsupported_rules=False)
        rule_dict['xmlhttp_third'] = AdblockRules(rules, use_re2=False, max_mem=1024*1024*1024, supported_options=['third-party', 'xmlhttprequest', 'domain', 'subdocument'], skip_unsupported_rules=False)
        rule_dict['third'] = AdblockRules(rules, use_re2=False, max_mem=1024*1024*1024, supported_options=['third-party', 'domain', 'subdocument'], skip_unsupported_rules=False)
        rule_dict['domain'] = AdblockRules(rules, use_re2=False, max_mem=1024*1024*1024, supported_options=['domain', 'subdocument'], skip_unsupported_rules=False)
        filterlist_rules[fname] = rule_dict

    return filterlists, filterlist_rules


def match_url(domain_top_level, current_domain, current_url, resource_type, rules_dict):

    """
    Function to match node information with filter list rules.
    
    Args:
        domain_top_level: eTLD+1 of visited page.
        current_domain; Domain of request being labelled.
        current_url: URL of request being labelled.
        resource_type: Type of request being labelled (from content policy type).
        rules_dict: Dictionary of filter list rules.
    Returns:
        Label indicating whether the rule should block the node (True/False).
    """

    try:
        if domain_top_level == current_domain:
            third_party_check = False
        else:
            third_party_check = True

        if resource_type == 'sub_frame':
            subdocument_check = True
        else:
            subdocument_check = False

        if resource_type == 'script':
            if third_party_check:
                rules = rules_dict['script_third']
                options = {'third-party': True, 'script': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
            else:
                rules = rules_dict['script']
                options = {'script': True, 'domain': domain_top_level, 'subdocument': subdocument_check}

        elif resource_type == 'image' or resource_type == 'imageset':
            if third_party_check:
                rules = rules_dict['image_third']
                options = {'third-party': True, 'image': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
            else:
                rules = rules_dict['image']
                options = {'image': True, 'domain': domain_top_level, 'subdocument': subdocument_check}

        elif resource_type == 'stylesheet':
            if third_party_check:
                rules = rules_dict['css_third']
                options = {'third-party': True, 'stylesheet': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
            else:
                rules = rules_dict['css']
                options = {'stylesheet': True, 'domain': domain_top_level, 'subdocument': subdocument_check}

        elif resource_type == 'xmlhttprequest':
            if third_party_check:
                rules = rules_dict['xmlhttp_third']
                options = {'third-party': True, 'xmlhttprequest': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
            else:
                rules = rules_dict['xmlhttp']
                options = {'xmlhttprequest': True, 'domain': domain_top_level, 'subdocument': subdocument_check}

        elif third_party_check:
            rules = rules_dict['third']
            options = {'third-party': True, 'domain': domain_top_level, 'subdocument': subdocument_check}

        else:
            rules = rules_dict['domain']
            options = {'domain': domain_top_level, 'subdocument': subdocument_check}

        return rules.should_block(current_url, options)

    except Exception as e:
        print('Exception encountered', e)
        print('top url', domain_top_level)
        print('current url', current_domain)
        return False

def label_node_data(row, filterlists, filterlist_rules):

    """
    Function to label a node with filter lists. 
    
    Args:
        row: Row of node DataFrame.
        filterlists: List of filter list names.
        filterlist_rules: Dictionary of filter lists and their rules.
    Returns:
        data_label: Label for node (True/False).
    """

    try:
        top_domain = row['top_level_domain']
        url = row['name']
        domain = row['domain']
        resource_type = row['resource_type']
        data_label = False

        for fl in filterlists:
            if top_domain and domain:
                list_label = match_url(top_domain, domain, url, resource_type, filterlist_rules[fl])
                data_label = data_label | list_label
            else:
                data_label = "Error"
    except Exception as e:
        print('Error in node labelling:', e)
        data_label = "Error"

    return data_label


def label_nodes(df, filterlists, filterlist_rules):

    """
    Function to label nodes with filter lists. 
    
    Args:
        df: DataFrame of nodes.
        filterlists: List of filter list names.
        filterlist_rules: Dictionary of filter lists and their rules.
    Returns:
        df_nodes: DataFrame of labelled nodes.
    """

    df_nodes = df[(df['type'] != 'Storage') & (df['type'] != 'Element')].copy()
    df_nodes['resource_type'] = df_nodes['attr'].apply(get_resource_type)
    df_nodes['label'] = df_nodes.apply(label_node_data, filterlists=filterlists, filterlist_rules=filterlist_rules, axis=1)
    df_nodes = df_nodes[['visit_id', 'name', 'top_level_url', 'label']]

    return df_nodes


def label_data(df, filterlists, filterlist_rules):
    df_labelled = pd.DataFrame()

    try:
        df_nodes = df[df['graph_attr'] == "Node"]
        df_labelled = label_nodes(df_nodes, filterlists, filterlist_rules)
    except Exception as e:
        print("Error labelling:", e)

    return df_labelled