import pandas as pd
import json
import re
import numpy as np

from logger import LOGGER

def get_attr(cpt, tlu):

    """
    Function to process attributes for request nodes.

    Args:
        cpt: content policy type.
        tlu: top level URL.
    Returns:
        Attribute string.
    """

    record = {'content_policy_type': cpt, 'top_level_url': tlu}
    return json.dumps(record)


def convert_type(orig_type, attr):

    """
    Function to convert Request type to Script/Document types based on the content policy type attribute.

    Args:
        orig_type: Original type (Reuqest).
        attr: Attributes of request.
    Returns:
        new_type: Changed type.
    """

    attr = json.loads(attr)
    new_type = orig_type
    if attr["content_policy_type"] == "script":
        new_type = "Script"
    if attr["content_policy_type"] == "main_frame":
        new_type = "Document"
    return new_type


def get_key(v1, v2):

    """
    Function to get key to link request/response/redirect with same IDs.

    Args:
        v1: Visit ID.
        v2: Request ID.
    Returns:
        key
    """

    return str(int(v1)) + "_" + str(int(v2))


def process_attr(respattr):

    """
    Function to process response headers,

    Args:
        respattr: Response headers.
    Returns:
        Processed headers as attributes.
    """

    attr = {}
    try:
        respattr = json.loads(respattr)
        for item in respattr:
            if item[0] == 'Content-Length':
                attr["clength"] = int(item[1])
            if item[0] == 'Content-Type':
                attr["ctype"] = item[1]
        return json.dumps(attr)
    except:
        return None


def process_redirects(df):

    """
    Function to process redirect data.

    Args:
        df: DataFrame of merged redirect request/responses.
    Returns:
        edges: DataFrame representation of redirects.
    """

    header_list = df['respattr1'].append(
        df.iloc[-1:]['headers'], ignore_index=True)
    status_list = df['response_status_x'].append(
        df.iloc[-1:]['response_status_y'], ignore_index=True)
    edges = df[['visit_id', 'old_request_url',
                'new_request_url', 'top_level_url_x', 'reqattr2', 'time_stamp_x']]
    edges = edges.rename(columns={'old_request_url': 'src', 'time_stamp_x': 'time_stamp',
                                  'new_request_url': 'dst', 'reqattr2': 'reqattr', 'top_level_url_x' : 'top_level_url'})
    first_row = df.iloc[0]
    data = []
    new_entry = {'visit_id': first_row['visit_id'], 'src': first_row['top_level_url_x'],
                 'dst': first_row['old_request_url'], 'top_level_url': first_row['top_level_url_x'], 'reqattr': first_row['reqattr1'],
                 'time_stamp': first_row['time_stamp_x']}
    data.insert(0, new_entry)
    edges = pd.concat([pd.DataFrame(data), edges], ignore_index=True)
    edges['respattr'] = header_list
    edges['response_status'] = status_list
    return edges


def get_redirect_edges(df_requests, df_redirects, df_responses):

    """
    Function to build redirects edges.

    Args:
        df_requests: DataFrame representation of requests table in OpenWPM.
        df_redirects: DataFrame representation of redirects table in OpenWPM.
        df_responses: DataFrame representation of responses table in OpenWPM.
        is_webgraph: compute additional info for WebGraph mode
    Returns:
        df_redirect_edges: DataFrame representation of redirect edges.
        completed_ids: Request IDs of redirect edges.
    """

    df_reqheaders = df_requests[['visit_id', 'request_id',
                                 'url', 'headers', 'top_level_url', 'time_stamp']]
    df_red = df_redirects[['visit_id', 'old_request_id',
                           'old_request_url', 'new_request_url', 'headers', 'response_status']]

    x1 = pd.merge(df_red, df_reqheaders, left_on=['visit_id', 'old_request_id', 'old_request_url'],
                  right_on=['visit_id', 'request_id', 'url'])
    x2 = pd.merge(x1, df_requests, left_on=['visit_id', 'old_request_id', 'new_request_url'],
                  right_on=['visit_id', 'request_id', 'url'])
    x2 = x2.rename(columns={'headers_x': 'respattr1',
                            'headers_y': 'reqattr1', 'headers': 'reqattr2'})
    x3 = pd.merge(x2, df_responses, left_on=['visit_id', 'old_request_id', 'new_request_url'],
                  right_on=['visit_id', 'request_id', 'url'], how='outer')

    df_redirect_edges = x3.groupby(['visit_id', 'old_request_id'], as_index=False).apply(
        process_redirects).reset_index()
    df_redirect_edges = df_redirect_edges[[
        'visit_id', 'src', 'dst', 'top_level_url', 'reqattr', 'respattr', 'response_status', 'time_stamp']]
    df_redirect_edges['content_hash'] = "N/A"
    df_redirect_edges['post_body'] = np.nan
    df_redirect_edges['post_body_raw'] = np.nan

    completed_ids = x3['key_x'].unique().tolist()

    return df_redirect_edges, completed_ids


def process_call_stack(row):

    """
    Function to process callstacks to get callstack edge data.

    Args:
        row: Row of callstack DataFrame.
    Returns:
        edge_data: callstack edge data
    """

    cs_lines = row['call_stack'].split()
    urls = []
    new_urls = []
    min_len = 5
    for line in cs_lines:
        url_parsing = re.search("(?P<url>https?://[^\s:]+)", line)
        if url_parsing != None:
            urls.append(url_parsing.group("url"))
    urls = urls[::-1]
    urls = list(set(urls))
    for url in urls:
        if len(new_urls) == 0:
            new_urls.append(url)
        else:
            if new_urls[-1] != url:
                new_urls.append(url)
    edge_data = []
    if len(new_urls) > 1:
        for i in range(0, len(new_urls) - 1):
            src_cs = new_urls[i]
            dst_cs = new_urls[i+1]
            reqattr_cs = "CS"
            respattr_cs = "CS"
            status_cs = "CS"
            post_body_cs = "CS"
            post_body_raw_cs = "CS"
            edge_data.append([src_cs, dst_cs, reqattr_cs, respattr_cs, status_cs,
                              row['time_stamp'], row['visit_id'], row['content_hash'], post_body_cs, post_body_raw_cs])
    if len(new_urls) > 0:
        edge_data.append([new_urls[-1], row['name'], row['reqattr'], row['respattr'],
                          row['response_status'], row['time_stamp'], row['visit_id'], row['content_hash'], row['post_body'], row['post_body_raw']])

    return edge_data


def get_cs_edges(df_requests, df_responses, call_stacks):

    """
    Function to build callstack edges.

    Args:
        df_requests: DataFrame representation of requests table in OpenWPM.
        df_responses: DataFrame representation of responses table in OpenWPM.
        call_stacks: DataFrame representation of call_stacks table in OpenWPM.
    Returns:
        df_cs_edges: DataFrame representation of callstack edges.
        completed_ids: Request IDs of call stack edges.
    """

    df_merge = pd.merge(df_requests, df_responses, on=[
                        "visit_id", "request_id"], how="inner")
    call_stack_nodup = call_stacks[[
        'visit_id', 'request_id', 'call_stack']].drop_duplicates().copy()
    df_merge = pd.merge(df_merge, call_stack_nodup, on=[
                        "visit_id", "request_id"], how="inner")
    df_merge = df_merge[['visit_id', 'url_x', 'top_level_url', 'headers_x',
                         'headers_y', 'time_stamp_x', 'response_status', 'post_body', 'post_body_raw', 'content_hash', 'call_stack', 'key_x']]
    df_merge = df_merge.rename(columns={'url_x': 'name', 'headers_x': 'reqattr', 'headers_y': 'respattr',
                                        'time_stamp_x': 'time_stamp', 'key_x': 'key'})
    
    # return empty outputs if empty
    if len(df_merge) == 0:
        return pd.DataFrame(), []
      
    df_merge['cs_edges'] = df_merge.apply(process_call_stack, axis=1)
    df = df_merge[['top_level_url', 'cs_edges']]
    df = df.explode('cs_edges').dropna()
    df['src'] = df['cs_edges'].apply(lambda x: x[0])
    df['dst'] = df['cs_edges'].apply(lambda x: x[1])
    df['reqattr'] = df['cs_edges'].apply(lambda x: x[2])
    df['respattr'] = df['cs_edges'].apply(lambda x: x[3])
    df['response_status'] = df['cs_edges'].apply(lambda x: x[4])
    df['time_stamp'] = df['cs_edges'].apply(lambda x: x[5])
    df['visit_id'] = df['cs_edges'].apply(lambda x: x[6])
    df['content_hash'] = df['cs_edges'].apply(lambda x: x[7])
    df['post_body'] = df['cs_edges'].apply(lambda x: x[8])
    df['post_body_raw'] = df['cs_edges'].apply(lambda x: x[9])
    df_cs_edges = df.drop(columns=['cs_edges']).reset_index()
    del df_cs_edges['index']

    completed_ids = df_merge['key'].unique().tolist()

    return df_cs_edges, completed_ids


def get_normal_edges(df_requests, df_responses, completed_ids):

    """
    Function to build edges that are not redirect edges.

    Args:
        df_requests: DataFrame representation of requests table in OpenWPM.
        df_responses: DataFrame representation of responses table in OpenWPM.
        completed_ids: Request IDs that were redirect or call stack edges.
    Returns:
        df_normal_edges: DataFrame representation of non-redirect edges.
    """

    df_remaining = df_requests[~df_requests['key'].isin(completed_ids)]
    df_remaining = pd.merge(df_remaining, df_responses, on=['key'])
    df_normal_edges = df_remaining[['visit_id_x', 'top_level_url', 'url_x', 'headers_x',
                                    'headers_y', 'response_status', 'post_body', 'post_body_raw', 'time_stamp_x', 'content_hash']]
    df_normal_edges = df_normal_edges.rename(columns={'visit_id_x': 'visit_id', 'top_level_url': 'src', 'url_x': 'dst',
                                                      'headers_x': 'reqattr', 'headers_y': 'respattr',
                                                      'time_stamp_x': 'time_stamp'})
    df_normal_edges['top_level_url'] = df_normal_edges['src']

    return df_normal_edges


def build_request_components(df_requests, df_responses, df_redirects, call_stacks, is_webgraph: bool):

    """
    Function to extract HTTP nodes/edges.

    Args:
        df_requests: DataFrame representation of requests table in OpenWPM.
        df_responses: DataFrame representation of responses table in OpenWPM.
        df_redirects: DataFrame representation of redirects table in OpenWPM.
        call_stacks: DataFrame representation of call_stacks table in OpenWPM.
        is_webgraph: compute additional info for WebGraph mode
    Returns:
        df_request_nodes: DataFrame representation of request nodes.
        df_request_edges: DataFrame representation of request edges.
    """

    df_request_nodes = pd.DataFrame()
    df_request_edges = pd.DataFrame()

    try:

        """Function to build HTTP nodes and edges from the OpenWPM HTTP data."""
        df_requests['key'] = df_requests[['visit_id', 'request_id']].apply(
            lambda x: get_key(*x), axis=1)
        df_responses['key'] = df_responses[['visit_id', 'request_id']].apply(
            lambda x: get_key(*x), axis=1)

        df_requests['type'] = 'Request'
        df_requests['attr'] = df_requests[['resource_type', 'top_level_url']].apply(
            lambda x: get_attr(*x), axis=1)
        # Request nodes. To be inserted
        df_request_nodes = df_requests[[
            'visit_id', 'url', 'type', 'top_level_url', 'attr']].drop_duplicates().copy()
        df_request_nodes['type'] = df_requests[['type', 'attr']].apply(
            lambda x: convert_type(*x), axis=1)
        df_request_nodes = df_request_nodes.rename(columns={'url': 'name'})

        # Redirect edges. To be inserted
        if len(df_redirects) > 0 and is_webgraph:
            df_redirects['old_request_id'] = df_redirects['old_request_id'].apply(
                lambda x: int(x))
            df_redirects['key'] = df_redirects[['visit_id', 'old_request_id']].apply(
                lambda x: get_key(*x), axis=1)
            df_redirect_edges, completed_ids_red = get_redirect_edges(
                df_requests, df_redirects, df_responses)
        else:
            completed_ids_red = []
            df_redirect_edges = pd.DataFrame()

        # CS edges
        df_cs_edges, completed_ids_cs = get_cs_edges(
            df_requests, df_responses, call_stacks)

        # Other edges. To be inserted
        completed_ids = set(completed_ids_red + completed_ids_cs)
        df_normal_edges = get_normal_edges(
            df_requests, df_responses, completed_ids)

        df_request_edges = pd.concat(
            [df_redirect_edges, df_cs_edges, df_normal_edges]).reset_index()
        del df_request_edges['index']
        df_request_edges['action'] = "N/A"

    except Exception as e:
        LOGGER.warning("Error in request_components:", exc_info=True)

    return df_request_nodes, df_request_edges
