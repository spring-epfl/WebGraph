import re
import json
import traceback

import pandas as pd


def check_tag_dict(row, tagdict):
        
    """Function to check if an element is inline."""
    vid = row['visit_id']
    url = row['script_url']

    try:
        if tagdict.get(vid).get(url):
            return "Present"
        else:
            return "Absent"
    except Exception as e:
        return "Absent"
    
def get_inline_name(row, tagdict):
        
    """Function to set name of an inline script using its location in a file."""
    vid = row['visit_id']
    url = row['script_url']
    line = row['script_line']

    try:
        name = ""
        line_tags = tagdict.get(vid).get(url)
        if line_tags:
            for item in line_tags:
                if line >= item[0] and line < item[1]:
                    name = "inline_" + url + "_" + str(item[0]) + "_" + str(item[1])
                    break
            if len(name) == 0:
                name = "inline_" + url + "_0_0"
        else:
            name = "inline_" + url + "_0_0"
        return name
    except Exception as e:
        name = "inline_" + url + "_0_0"
        return name
    
def convert_attr(row):
        
    """Function to convert attribute of an element."""
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
        print(e)
        return json.dumps(attr)

def convert_subtype(x):
        
    """Function to convert subtype of an element."""

    try:
        return json.loads(x)[0]
    except Exception as e:
        return ""

def get_eval_name(script, line):
        
    """Function to get name of an eval node based on the location in a script."""

    try:
        return line.split()[3] + "_" + line.split()[1] + "_" + script
    except:
        return "Eval_error" 

def get_tag(record, key):
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
    
def get_attr_action(symbol):
        
    """Function to find attribute modification (new, deleted, changed)."""

    try:
        result = re.search('.attr_(.*)', symbol)
        return "attr_" + result.group(1)
    except Exception as e:
        return ""

def find_parent_elem(src_elements, df_element):
    
    """Function to find parent elements."""
    
    src_elements['new_attr'] = src_elements['attributes'].apply(get_tag, key="fullopenwpm")
    df_element['new_attr'] = df_element['attr'].apply(get_tag, key="openwpm")
    result = src_elements.merge(df_element[['new_attr', 'name']], on='new_attr', how='left')
    return result


def find_modified_elem(df_element, df_javascript):
    
    """Function to find modified elements where the attribute has changed."""
    
    df_javascript['new_attr'] = df_javascript['attributes'].apply(get_tag, key="openwpm")
    df_element['new_attr'] = df_element["attr"].apply(get_tag, key="openwpm")
    result = df_javascript.merge(df_element[['new_attr', 'name']], on='new_attr', how='left')
    return result


def build_html_components(df_javascript):
    """Function to create HTML nodes and edges in WebGraph."""

    df_js_nodes = pd.DataFrame()
    df_js_edges = pd.DataFrame()

    try:
        #Find all created elements
        created_elements = df_javascript[df_javascript['symbol'] == 'window.document.createElement'].copy()
        created_elements['name'] = created_elements.index.to_series().apply(lambda x: "Element_" + str(x))
        created_elements['type'] = 'Element'

        created_elements['subtype_list'] = created_elements['arguments'].apply(convert_subtype)
        created_elements['attr'] = created_elements.apply(convert_attr, axis=1)
        created_elements['action'] = 'create'

        #Created Element nodes and edges (to be inserted)
        df_element_nodes = created_elements[['visit_id', 'name', 'top_level_url', 'type', 'attr']]
        df_create_edges = created_elements[['visit_id', 'script_url', 'name', 'top_level_url', 'action', 'time_stamp']]
        df_create_edges = df_create_edges.rename(columns={'script_url' : 'src', 'name' : 'dst'})
        
        src_elements = df_javascript[(df_javascript['symbol'].str.contains("Element.src")) & (df_javascript['operation'].str.contains('set'))].copy()
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
        print("Error in build_html_components:", e)
        traceback.print_exc()
        return df_js_nodes, df_js_edges

    return df_js_nodes, df_js_edges
