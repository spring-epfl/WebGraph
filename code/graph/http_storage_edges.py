import pandas as pd
from .cookies import *
import json
import re
import multidict
from .utils import *

from logger import LOGGER

def parse_setcookie_header(cookie_header):

    """
    Function to parse Set-Cookie HTTP header.

    Args:
        cookie_header: Set-Cookie HTTP header
    Returns:
        cookie_list: List of cookie attributes of set cookies.
    """

    cookie_list = []
    try:
        cookie = BaseCookie(cookie_header)
        for k in cookie.keys():
            name = k
            value = cookie[k].value
            expires = cookie[k]['expires'] if 'expires' in cookie[k] else None
            path = cookie[k]['path'] if 'path' in cookie[k] else None
            domain = cookie[k]['domain'] if 'domain' in cookie[k] else None
            max_age = cookie[k]['max-age'] if 'max-age' in cookie[k] else None
            httponly = True if cookie[k]['httponly'] is True else False
            secure = True if cookie[k]['secure'] is True else False
            samesite = cookie[k]['samesite'] if 'samesite' in cookie[k] else None

            cookie_dict = {"name" : name, "value" : value, "expires" : expires, "path" : path,
                "domain" : domain, "max_age" : max_age, "httponly" : httponly, "secure" : secure, "samesite" : samesite}
            cookie_list.append(cookie_dict)
    except Exception as e:
        return cookie_list
    return cookie_list

def parse_cookie_header(cookie_header):

    """
    Function to parse Cookie HTTP header.

    Args:
        cookie_header: Cookie HTTP header
    Returns:
        cookie_list: List of read cookies from Cookie header.
    """

    cookie_list = []
    try:
        cookie = BaseCookie(cookie_header)
        for k in cookie.keys():
            name = k
            value = cookie[k].value
            cookie_dict = {"name" : name, "value" : value}
            cookie_list.append(cookie_dict)
    except Exception as e:
        return cookie_list
    return cookie_list

def get_cookie_details(row):

    """
    Function to get cookie details from request/response headers.

    Args:
        row: Row data of a particular HTTP request.
    Returns:
        cookie_details: List of cookie details for set/read cookies.
    """

    cookie_details = []
    try:
        reqattr = row['reqattr']
        respattr = row['respattr']
        dst = row['dst']
        visit_id = row['visit_id']
        time_stamp = row['time_stamp']
        headers = []
        if not pd.isna(reqattr):
            headers += json.loads(reqattr)
        if not pd.isna(respattr):
            headers += json.loads(respattr)
        #headers = json.loads(reqattr) + json.loads(respattr)
        header_dict = multidict.MultiDict(headers)

        if "Cookie" in header_dict.keys():
            cookie_list = parse_cookie_header(header_dict["Cookie"])
            for cookie in cookie_list:
                cookie_details.append([dst, cookie["name"], "get", json.dumps(cookie), visit_id, time_stamp])
        if "Set-Cookie" in header_dict.keys():
            cookie_list = parse_setcookie_header(header_dict["Set-Cookie"])
            for cookie in cookie_list:
                cookie_details.append([dst, cookie["name"], "set", json.dumps(cookie), visit_id, time_stamp])
        if "set-cookie" in header_dict.keys():
            cookie_list = parse_setcookie_header(header_dict["set-cookie"])
            for cookie in cookie_list:
                cookie_details.append([dst, cookie["name"], "set", json.dumps(cookie), visit_id, time_stamp])
    except Exception as e:
        LOGGER.warning("Error in http_cookies: getting cookie details")

    return cookie_details

def build_http_cookie_components(df_http_edges, df_http_nodes):

    """
    Function to extract HTTP cookie nodes (cookies set via HTTP headers).

    Args:
        df_http_edges: DataFrame representation of HTTP request edges
        df_http_nodes: DataFrame representation of HTTP request nodes
    Returns:
        df_http_cookie_nodes: DataFrame representation of HTTP cookie nodes
        df_http_cookie_edges: DataFrame representation of HTTP cookie edges
    """

    df_http_cookie_nodes = pd.DataFrame()
    df_http_cookie_edges = pd.DataFrame()

    try:

        df_cookies = df_http_edges[(df_http_edges['respattr'].str.contains('Set-Cookie')) \
                                                | (df_http_edges['respattr'].str.contains('set-cookie'))
                                                            | (df_http_edges['reqattr'].str.contains('Cookie'))].copy()
        if len(df_cookies) > 0:
            df_cookies['cookie_details'] = df_cookies.apply(get_cookie_details, axis=1)
            df_cookies = df_cookies[['cookie_details']].explode('cookie_details').dropna()
            df_cookies['src'] = df_cookies['cookie_details'].apply(lambda x: x[0])
            df_cookies['dst'] = df_cookies['cookie_details'].apply(lambda x: x[1])
            df_cookies['action'] = df_cookies['cookie_details'].apply(lambda x: x[2])
            df_cookies['attr'] = df_cookies['cookie_details'].apply(lambda x: x[3])
            df_cookies['visit_id'] = df_cookies['cookie_details'].apply(lambda x: x[4])
            df_cookies['time_stamp'] = df_cookies['cookie_details'].apply(lambda x: x[5])
            df_cookies = df_cookies.merge(df_http_nodes[['visit_id', 'name', 'top_level_url']], left_on=['visit_id', 'src']
                                , right_on=['visit_id', 'name'])
            df_cookies['domain'] = df_cookies['src'].apply(get_domain)
            df_cookies['cookie_key'] = df_cookies[['dst', 'domain']].apply(
                lambda x: get_cookiedom_key(*x), axis=1)

            #To be inserted
            df_http_cookie_nodes = df_cookies[["visit_id", "cookie_key", "top_level_url", "domain"]].copy().drop_duplicates()
            df_http_cookie_nodes = df_http_cookie_nodes.rename(columns={'cookie_key' : 'name'})
            df_http_cookie_nodes['type'] = "Storage"
            df_http_cookie_nodes['attr'] = '{"cookie_type": "HTTPCookie"}'

            df_http_cookie_edges = df_cookies.drop(columns=['cookie_details', 'dst']).reset_index()
            del df_http_cookie_edges['index']
            df_http_cookie_edges = df_http_cookie_edges.rename(columns={'cookie_key' : 'dst'})
            df_http_cookie_edges['reqattr'] = "N/A"
            df_http_cookie_edges['respattr'] = "N/A"
            df_http_cookie_edges['response_status'] = "N/A"
            df_http_cookie_edges['post_body'] = np.nan
            df_http_cookie_edges['post_body_raw'] = np.nan

    except Exception as e:
        LOGGER.warning("Error in http_cookie_components:", exc_info=True)

    return df_http_cookie_nodes, df_http_cookie_edges
