import json
import pandas as pd

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
        return None

      
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
    except Exception:
        LOGGER.warning('Error in node labelling:', exc_info=True)
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
        LOGGER.warning("Error labelling:", exc_info=True)

    return df_labelled
