import re
import json
import traceback

import pandas as pd

from logger import LOGGER

def convert_attr(row):

    """
    Function to create attributes for created elements.

    Args:
        row: Row of created elements DataFrame.
    Returns:
        attr: JSON string of attributes for created elements.
    """

    attr = {}
    try:
        attr["openwpm"] = json.loads(row['attributes'])["0"]["openwpm"]
        attr['subtype'] = row['subtype_list']
        if row['script_loc_eval'] != "":
            attr['eval'] = True
        else:
            attr['eval'] = False
        attr = json.dumps(attr)
        return attr
    except Exception as e:
        LOGGER.warning("[ convert_attr ] : ERROR - ", exc_info=True)
        return json.dumps(attr)

def convert_subtype(arguments):

    """
    Function to obtain subtype of an element.

    Args:
        arguments: arguments column of javascript table for a created element.
    Returns:
        Sub-type fo created element (or emptry string in case of errors).
    """

    try:
        return json.loads(x)[0]
    except Exception as e:
        return ""

def get_tag(record, key):

    """
    Function to obtain the openwpm tag value.

    Args:
        record: Record to check tags.
        key: Key to get correct tag value.
    Returns:
        Tag value.
    """

    try:
        val = json.loads(record)

        if key == "fullopenwpm":
            openwpm = val.get("0").get("openwpm")
            return str(openwpm)
        else:
            return str(val.get(key))
    except Exception as e:
        return ""
    return ""


def find_parent_elem(src_elements, df_element):

    """
    Function to find parent element of .src JS elements.

    Args:
        src_element: DataFrame representation of src elements.
        df_element: DataFrame representation of created elements.
    Returns:
        result: Merged DataFrame representation linking created elements with src elements.
    """

    src_elements['new_attr'] = src_elements['attributes'].apply(get_tag, key="fullopenwpm")
    df_element['new_attr'] = df_element['attr'].apply(get_tag, key="openwpm")
    result = src_elements.merge(df_element[['new_attr', 'name']], on='new_attr', how='left')
    return result


def build_html_components(df_javascript):

    """
    Function to create HTML nodes and edges. This is limited since we
    don't capture all HTML behaviors -- we look at createElement and src JS calls.

    Args:
        df_javascript: DataFrame representation of OpenWPM's javascript table.
    Returns:
        df_js_nodes: DataFrame representation of HTML nodes
        df_js_edges: DataFrame representation of HTML edges
    """

    df_js_nodes = pd.DataFrame()
    df_js_edges = pd.DataFrame()

    try:
        #Find all created elements
        created_elements = df_javascript[df_javascript['symbol'] == 'window.document.createElement'].copy()

        df_element_nodes = pd.DataFrame(columns=['visit_id', 'name', 'top_level_url', 'type', 'attr'])

        if len(created_elements) > 0:
            created_elements['name'] = created_elements.index.to_series().apply(lambda x: "Element_" + str(x))
            created_elements['type'] = 'Element'

            created_elements['subtype_list'] = created_elements['arguments'].apply(convert_subtype)
            created_elements['attr'] = created_elements.apply(convert_attr, axis=1)
            created_elements['action'] = 'create'

            #Created Element nodes and edges (to be inserted)
            df_element_nodes = created_elements[['visit_id', 'name', 'top_level_url', 'type', 'attr']]
            df_create_edges = created_elements[['visit_id', 'script_url', 'name', 'top_level_url', 'action', 'time_stamp']]
            df_create_edges = df_create_edges.rename(columns={'script_url' : 'src', 'name' : 'dst'})
        else:
            df_create_edges = pd.DataFrame()
            df_element_nodes = pd.DataFrame()
           
        src_elements = df_javascript[(df_javascript['symbol'].str.contains("Element.src")) & (df_javascript['operation'].str.contains('set'))].copy()
        
        if len(src_elements) > 0:
            src_elements['type'] = "Request"
            src_elements = find_parent_elem(src_elements, df_element_nodes)
            src_elements['action'] = "setsrc"

            #Src Element nodes and edges (to be inserted)
            df_src_nodes = src_elements[['visit_id', 'value', 'top_level_url', 'type', 'attributes']].copy()
            df_src_nodes = df_src_nodes.rename(columns={'value': 'name', 'attributes': 'attr'})
            df_src_nodes = df_src_nodes.dropna(subset=["name"])
        

            df_src_edges = src_elements[['visit_id', 'name', 'value', 'top_level_url', 'action', 'time_stamp']]
            df_src_edges = df_src_edges.dropna(subset=["name"])
            df_src_edges = df_src_edges.rename(columns={'name': 'src', 'value': 'dst'})
        
            df_js_nodes = pd.concat([df_element_nodes, df_src_nodes]).drop_duplicates()
            df_js_nodes = df_js_nodes.drop(columns=['new_attr'])
            df_js_edges = pd.concat([df_create_edges, df_src_edges])
        
            df_js_edges['reqattr'] = "N/A"
            df_js_edges['respattr'] = "N/A"
            df_js_edges['response_status'] = "N/A"
            df_js_edges['attr'] = "N/A"

    except Exception as e:
        LOGGER.warning("Error in build_html_components:", exc_info=True)
        return df_js_nodes, df_js_edges

    return df_js_nodes, df_js_edges
