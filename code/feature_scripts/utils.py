import pandas as pd
import json
import networkx as nx
import re
#import plyvel
from openwpm_utils import domain as du
from six.moves.urllib.parse import urlparse, parse_qs
from sklearn import preprocessing
from yaml import load, dump
import numpy as np
import traceback

import graph as gs
import base64
import hashlib

def has_ad_keyword(node, G):

  keyword_raw = ["ad", "ads", "advert", "popup", "banner", "sponsor", "iframe", "googlead", "adsys", "adser", "advertise", "redirect",
                 "popunder", "punder", "popout", "click", "track", "play", "pop", "prebid", "bid", "pb.min", "affiliate", "ban",
                 "delivery", "promo","tag", "zoneid", "siteid", "pageid", "size", "viewid", "zone_id", "google_afc" , "google_afs"]
  has_ad_keyword = 0
  node_type = G.nodes[node]['type']
  if node_type != "Element" and node_type != "Storage":
    for key in keyword_raw:
        key_matches = [m.start() for m in re.finditer(key, node, re.I)]
        for key_match in key_matches:
          has_ad_keyword = 1
          break
        if has_ad_keyword == 1:
          break
  return has_ad_keyword

def ad_keyword_ascendants(node, G):

  keyword_raw = ["ad", "ads", "advert", "popup", "banner", "sponsor", "iframe", "googlead", "adsys", "adser", "advertise", "redirect",
                 "popunder", "punder", "popout", "click", "track", "play", "pop", "prebid", "bid", "pb.min", "affiliate", "ban", "delivery",
                 "promo","tag", "zoneid", "siteid", "pageid", "size", "viewid", "zone_id", "google_afc" , "google_afs"]

  ascendant_has_ad_keyword = 0
  ascendants = nx.ancestors(G, node)
  for ascendant in ascendants:
    try:
      node_type = nx.get_node_attributes(G, 'type')[ascendant]
      if node_type != "Element" and node_type != "Storage":
        for key in keyword_raw:
          key_matches = [m.start() for m in re.finditer(key, ascendant, re.I)]
          for key_match in key_matches:
            ascendant_has_ad_keyword = 1
            break
          if ascendant_has_ad_keyword == 1:
            break
      if ascendant_has_ad_keyword == 1:
            break
    except:
      continue
  return ascendant_has_ad_keyword

def get_num_attr_changes(df_modified_attr, G):

  num_attr_changes = 0
  source_list = df_modified_attr['src'].tolist()
  for source in source_list:
    if nx.get_node_attributes(G, 'type')[source] == 'Script':
      num_attr_changes += 1
  return num_attr_changes

def find_modified_storage(df_target):

  df_modedges = pd.DataFrame()
  df_copy = df_target.copy().sort_values(by=['time_stamp'])
  df_copy = df_copy.reset_index()
  set_node = df_copy.iloc[[0]][['src','dst']]
  modify_nodes = df_copy.drop([0], axis=0)[['src','dst']]
  if len(modify_nodes) > 0:
    df_merged = pd.merge(set_node, modify_nodes, on='dst')
    df_modedges = df_merged[['src_x', 'src_y', 'dst']].drop_duplicates()
    df_modedges.columns = ['src', 'dst', 'attr']
    df_modedges = df_modedges.groupby(['src', 'dst'])['attr'].apply(len).reset_index()
  return df_modedges

def get_cookieval(attr):

  try:
    attr = json.loads(attr)
    if 'value' in attr:
      return attr['value']
    else:
      return None
  except:
    return None

def get_cookiename(attr):

  try:
    attr = json.loads(attr)
    if 'name' in attr:
      return attr['name']
    else:
      return None
  except:
    return None

def get_redirect_depths(df_graph):

  dict_redirect = {}

  try:

    http_status = [300, 301, 302, 303, 307, 308]
    #http_status = [str(x) for x in http_status]

    df_redirect = df_graph[df_graph['response_status'].isin(http_status)]
    G_red = gs.build_networkx_graph(df_redirect)

    for n in G_red.nodes():
      dict_redirect[n] = 0
      dfs_edges = list(nx.edge_dfs(G_red, source=n, orientation='reverse'))
      ct = 0
      depths = []
      if len(dfs_edges) == 1:
        dict_redirect[n] = 1
      if len(dfs_edges) >= 2:
        ct += 1
        for i in range(1, len(dfs_edges)):
          if dfs_edges[i][1] != dfs_edges[i-1][0]:
            depths.append(ct)
            ct = 1
          else:
            ct += 1
        depths.append(ct)
        if len(depths) > 0:
          dict_redirect[n] = max(depths)

    return dict_redirect

  except Exception as e:
    print("Error in redirect:", e)
    return dict_redirect

def find_urls(df):
  src_urls = df['src'].tolist()
  dst_urls = df['dst'].tolist()
  return list(set(src_urls + dst_urls))

def check_full_cookie(cookie_value, dest):
  return True if len([item for item in cookie_value if item in dest and len(item) > 3]) > 0 else False

def check_partial_cookie(cookie_value, dest):
  for value in cookie_value:
      split_cookie = re.split(r'\.+|;+|]+|\!+|\@+|\#+|\$+|\%+|\^+|\&+|\*+|\(+|\)+|\-+|\_+|\++|\~+|\`+|\@+=|\{+|\}+|\[+|\]+|\\+|\|+|\:+|\"+|\'+|\<+|\>+|\,+|\?+|\/+', value)
      return True if len([item for item in split_cookie if item in dest and len(item) > 3]) > 0 else False
  return False

def check_base64_cookie(cookie_value, dest):
  return True if len([item for item in cookie_value if base64.b64encode(item.encode('utf-8')).decode('utf8') in dest and len(item) > 3]) > 0 else False


def check_md5_cookie(cookie_value, dest):
  return True if len([item for item in cookie_value if hashlib.md5(item.encode('utf-8')).hexdigest() in dest and len(item) > 3]) > 0 else False


def check_sha1_cookie(cookie_value, dest):
  return True if len([item for item in cookie_value if hashlib.sha1(item.encode('utf-8')).hexdigest() in dest and len(item) > 3]) > 0 else False

def check_full_cookie_set(cookie_value, dest):
  if (len(cookie_value) > 3) and (cookie_value in dest):
    return True
  else:
    return False

def check_partial_cookie_set(cookie_value, dest):
  split_cookie = re.split(r'\.+|;+|]+|\!+|\@+|\#+|\$+|\%+|\^+|\&+|\*+|\(+|\)+|\-+|\_+|\++|\~+|\`+|\@+=|\{+|\}+|\[+|\]+|\\+|\|+|\:+|\"+|\'+|\<+|\>+|\,+|\?+|\/+', cookie_value)
  for item in split_cookie:
    if len(item) > 3 and item in dest:
      return True
  return False


def check_base64_cookie_set(cookie_value, dest):
  if (len(cookie_value) > 3) and (base64.b64encode(cookie_value.encode('utf-8')).decode('utf8') in dest):
    return True
  else:
    return False

def check_md5_cookie_set(cookie_value, dest):
  if (len(cookie_value) > 3) and (hashlib.md5(cookie_value.encode('utf-8')).hexdigest() in dest):
    return True
  else:
    return False

def check_sha1_cookie_set(cookie_value, dest):
  if (len(cookie_value) > 3) and (hashlib.sha1(cookie_value.encode('utf-8')).hexdigest() in dest):
    return True
  else:
    return False

def check_cookie_presence(http_attr, dest):

  check_value = False

  try:
    http_attr = json.loads(http_attr)

    for item in http_attr:
      if 'Cookie' in item[0]:
          cookie_pairs = item[1].split(';')
          for cookie_pair in cookie_pairs:
            cookie_value = cookie_pair.strip().split('=')[1:]
            full_cookie = check_full_cookie(cookie_value, dest)
            partial_cookie = check_partial_cookie(cookie_value, dest)
            base64_cookie = check_base64_cookie(cookie_value, dest)
            md5_cookie = check_md5_cookie(cookie_value, dest)
            sha1_cookie = check_sha1_cookie(cookie_value, dest)
            check_value = check_value | full_cookie | partial_cookie | base64_cookie | md5_cookie | sha1_cookie
            if check_value:
              return check_value
  except:
    check_value = False
  return check_value


def find_indirect_edges(G, df_graph):

  df_edges = pd.DataFrame()

  try:

    storage_set = df_graph[(df_graph['action'] == 'set') | \
      (df_graph['action'] == 'set_js') | (df_graph['action'] \
       == 'set_storage_js')][['src','dst']]
    storage_get = df_graph[(df_graph['action'] == 'get') | \
      (df_graph['action'] == 'get_js') | (df_graph['action'] \
        == 'get_storage_js')][['src','dst']]

    #Nodes that set to nodes that get
    df_merged = pd.merge(storage_set, storage_get, on='dst')
    df_get_edges = df_merged[['src_x', 'src_y', 'dst']].drop_duplicates()
    if len(df_get_edges) > 0:
      df_get_edges.columns = ['src', 'dst', 'attr']
      df_get_edges['cookie'] = df_get_edges['attr']
      df_get_edges = df_get_edges.groupby(['src', 'dst'])['attr'].apply(len).reset_index()
      df_get_edges['type'] = 'set_get'
      df_edges = df_edges.append(df_get_edges, ignore_index=True)


    #Nodes that set to nodes that modify
    all_storage_set = df_graph[(df_graph['action'] == 'set') | \
      (df_graph['action'] == 'set_js') | (df_graph['action'] == 'set_storage_js') \
      | (df_graph['action'] == 'remove_storage_js')]
    df_modified_edges = all_storage_set.groupby('dst').apply(find_modified_storage)
    if len(df_modified_edges) > 0:
      df_modified_edges['type'] = 'set_modify'
      df_edges = df_edges.append(df_modified_edges, ignore_index=True)


    #Nodes that set to URLs with cookie value
    df_set_url_edges = pd.DataFrame()
    df_cookie_set = df_graph[(df_graph['action'] == 'set') | \
                  (df_graph['action'] == 'set_js')].copy()
    df_cookie_set['cookie_val'] = df_cookie_set['attr'].apply(get_cookieval)
    cookie_values = list(set(df_cookie_set[~df_cookie_set['cookie_val'].isnull()]['cookie_val'].tolist()))
    
    df_nodes = df_graph[(df_graph['graph_attr'] == 'Node') & \
                        ((df_graph['type'] == 'Request') | \
                        (df_graph['type'] == 'Script') | \
                        (df_graph['type'] == 'Document'))]['name']
    urls = df_nodes.tolist()
    check_set_value = False

    for dest in urls:
      for cookie_value in cookie_values:
        full_cookie = check_full_cookie_set(cookie_value, dest)
        partial_cookie = check_partial_cookie_set(cookie_value, dest)
        base64_cookie = check_base64_cookie_set(cookie_value, dest)
        md5_cookie = check_md5_cookie_set(cookie_value, dest)
        sha1_cookie = check_sha1_cookie_set(cookie_value, dest)
        check_set_value = full_cookie | partial_cookie | base64_cookie | md5_cookie | sha1_cookie
        if check_set_value:
          src = df_cookie_set[df_cookie_set['cookie_val'] == cookie_value]['src'].iloc[0]
          dst = dest
          attr = 1
          df_set_url_edges = df_set_url_edges.append({'src' : src, 'dst' : dst, 'attr': attr}, ignore_index=True)

    if len(df_set_url_edges) > 0:
      df_set_url_edges = df_set_url_edges.groupby(['src', 'dst'])['attr'].apply(len).reset_index()
      df_set_url_edges['type'] = 'set_url'
      df_edges = df_edges.append(df_set_url_edges, ignore_index=True)

    #Nodes that get to URLs with cookie value
    df_http_requests = df_graph[(df_graph['reqattr'] != 'CS') & (df_graph['src'] != 'N/A') & (df_graph['action'] != 'CS') & (df_graph['graph_attr'] != 'EdgeWG')]
    df_http_requests_merge = pd.merge(left=df_http_requests, right=df_http_requests, how='inner', left_on=['visit_id','dst'], right_on=['visit_id', 'src'])
    df_http_requests_merge = df_http_requests_merge[df_http_requests_merge['reqattr_x'].notnull()]

    df_http_requests_merge['cookie_presence'] = df_http_requests_merge.apply(
        axis=1,
        func=lambda x: check_cookie_presence(x['reqattr_x'], x['dst_y'])
      )

    df_get_url_edges = df_http_requests_merge[df_http_requests_merge['cookie_presence'] == True][['src_x', 'dst_y', 'attr_x']]
    if len(df_get_url_edges) > 0:
      df_get_url_edges.columns = ['src', 'dst', 'attr']
      df_get_url_edges = df_get_url_edges.groupby(['src', 'dst'])['attr'].apply(len).reset_index()
      df_get_url_edges['type'] = 'get_url'
      df_edges = df_edges.append(df_get_url_edges, ignore_index=True)

  except Exception as e:
    traceback.print_exc()
    return df_edges

  return df_edges
