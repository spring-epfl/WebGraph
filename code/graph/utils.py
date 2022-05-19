from multiprocessing import Pool, cpu_count
# Imports
from unicodedata import decomposition
from tqdm.auto import tqdm
import json
import re
import tldextract
import numpy as np
import pandas as pd
import pymysql
from adblockparser import AdblockRules
import hashlib
from io import StringIO
from csv import writer
import base64

import pathlib as Path
from Levenshtein import distance
from urllib.parse import parse_qs
import urllib.parse as urlparse
tqdm.pandas()

requests = None

def build_disconnect_entity_map():

    disconnect_entity_map = {}

    with open("entities.json", "r") as f:
        entities = json.load(f)
        entities = entities['entities']

    for key in entities.keys():
        properties = entities[key]['properties']
        resources = entities[key]['resources']
        for p in properties:
            if p not in disconnect_entity_map:
                disconnect_entity_map[p] = key
        for r in resources:
            if r not in disconnect_entity_map:
                disconnect_entity_map[r] = key

    return disconnect_entity_map


def get_disconnect_entity(url, disconnect_entity_map):
    if url in disconnect_entity_map:
        return disconnect_entity_map[url]
    else:
        return url


def load_requests_table(database, host_name, user_name, password, ssl_ca, ssl_key, ssl_cert, port):
    conn = pymysql.connect(host=host_name, port=port, user=user_name, password=password,
                           database=database, ssl_ca=ssl_ca, ssl_key=ssl_key, ssl_cert=ssl_cert)
    global requests
    print('Loading requests dataset from server')
    requests = pd.read_sql(
        'select visit_id, time_stamp, url, headers from http_requests', conn, parse_dates=['time_stamp'])
    conn.close()

def build_duck_entity_map():

    duck_entity_map = {}

    with open("tracker-radar/build-data/generated/entity_map.json") as f:
        tracker_radar = json.load(f)

    for key in tracker_radar.keys():
        props = tracker_radar[key]['properties']
        for p in props:
            if p not in duck_entity_map:
                duck_entity_map[p] = key

    return duck_entity_map

def get_duck_duck_go_entity(url, duck_entity_map):
    
    if url in duck_entity_map:
        return duck_entity_map[url]
    else:
        return url


def process_cookie_call_stack(cs):
    urls = []
    cs_lines = cs.split()

    for line in cs_lines:
        components = re.split('[@;]', line)
        if len(components) >= 2:
            urls.append(components[1].rsplit(":", 2)[0])

    try:
        # The URL at the top of the non-reversed call stack will the final calling script URL
        return urls[0]
    except Exception as e:
        print("Exception while handling following call stack: "+cs)


def process_cookie_js_set(cookie_info):
    val_dict = {}
    cookie_info_parts = cookie_info.split(";")
    if len(cookie_info_parts) > 0:
        cnv = cookie_info_parts[0].split('=')
        if len(cnv) == 2:
            val_dict['name'] = cnv[0].strip()
            val_dict['value'] = cnv[1].strip()
        for ci in cookie_info_parts[1:]:
            params = ci.split('=')
            if len(params) == 2:
                val_dict[params[0].strip()] = params[1].strip()
    return [val_dict]


def process_cookie_js_get(cookie_info):
    val_list = []
    try:
        cookie_info_parts = cookie_info.split(";")
        if len(cookie_info_parts) > 0:
            for info in cookie_info_parts:
                cnv = info.split('=')
                if len(cnv) == 2:
                    val_dict = {
                        'name': cnv[0].strip(), 'value': cnv[1].strip()}
                    val_list.append(val_dict)
    except Exception as e:
        print("Exception in processing cookie get value:", e, cookie_info)
    return val_list


def process_cookie_js(row):
    val_dict = {}
    if row['operation'] == 'get':
        val_dict = process_cookie_js_get(row['value'])
    elif row['operation'] == 'set':
        val_dict = process_cookie_js_set(row['value'])
    return val_dict


def get_cookie_value(row):
    value = 'N/A'
    try:
        return row['value_processed'].get('value')
    except Exception as e:
        x = 1
        print("Error in cookie key:", e)
    return value


def get_cookie_key(row):
    key = 'N/A'
    try:
        return row['value_processed'].get('name')
    except Exception as e:
        print("Error in cookie key:", e)
    return key

def get_cookiedom_key(name, domain):

    try:
        return name + '|$$|' + domain
    except:
        return name
    return name

def get_domain(url):
    try:
        if (isinstance(url, list)):
            domains = []
            for u in url:
                u = tldextract.extract(u)
                domains.append(u.domain+"."+u.suffix)
            return domains
        else:
            u = tldextract.extract(url)
            return u.domain+"."+u.suffix
    except:
        return None

def get_entity(url, disconnect_entity_map, duck_entity_map):

    if (isinstance(url, list)):
        entities = [get_duck_duck_go_entity(x, duck_entity_map) for x in domains]
        entities = [get_disconnect_entity(x, disconnect_entity_map) for x in entities]
        return entities
    else:
        entity = get_duck_duck_go_entity(url, duck_entity_map)
        entity = get_disconnect_entity(entity, disconnect_entity_map)
        return entity

def find_party(row):
    try:
        tlu = row['top_level_url']
        tld = get_domain(tlu)
        script_domain = row['script_domain']
        if tld and script_domain:
            tld_entity = get_entity(tld)
            script_entity = get_entity(script_domain)
            if tld_entity != script_entity:
                return 'third'
            else:
                return 'first'
        else:
            return 'N/A'
    except Exception as e:
        print('Error in party finding:', e)


def get_context(row):
    document_url = row['document_url']
    top_level_url = row['top_level_url']

    if document_url and top_level_url:
        document_entity = get_domain(document_url)
        top_level_entity = get_domain(top_level_url)
        if document_entity != top_level_entity:
            return 'third'
        else:
            return 'first'
    return 'N/A'

def get_original_cookie_setters(df):

    df_owners = {}
    df.sort_values('time_stamp', ascending=False, inplace=True)
    grouped = df.groupby(['visit_id', 'dst'])
    rows_added = 0

    for name, group in grouped: #tqdm(grouped, total=len(grouped), desc='Progress: Cookie Owners', leave=True, position=1):
        if len(group) > 0:
            name_dict = {'visit_id': name[0], 'dst': name[1]}
            final_dict = {**name_dict, **group.iloc[0].to_dict()}
            df_owners[rows_added] = final_dict
            rows_added += 1

    if (rows_added > 0):
        df_owners = pd.DataFrame.from_dict(df_owners, "index")
        df_owners = df_owners[['visit_id', 'dst', 'src', 'time_stamp']]
        df_owners = df_owners.rename(
            columns={"dst" : "name", "src": "setter", "time_stamp": "setting_time_stamp"})
        return df_owners
    else:
        return pd.DataFrame(columns=['visit_id', 'name', 'setter', 'setting_time_stamp'])


def get_original_js_cookie_setters(df):
    
    df_owners = {}
    df.sort_values('time_stamp', ascending=False, inplace=True)
    grouped = df.groupby(['visit_id', 'name'])
    rows_added = 0

    for name, group in grouped: #tqdm(grouped, total=len(grouped), desc='Progress: Cookie Owners', leave=True, position=1):
        group = group[group.action.str.contains('set')]
        if len(group) > 0:
            name_dict = {'visit_id': name[0], 'name': name[1]}
            final_dict = {**name_dict, **group.iloc[0].to_dict()}
            df_owners[rows_added] = final_dict
            rows_added += 1

    if (rows_added > 0):
        return pd.DataFrame.from_dict(df_owners, "index")
    else:
        return pd.DataFrame(columns=['visit_id', 'name', 'setter', 'setting_time_stamp'])


def find_conflict_type(row):
    conflict = ''
    if (row['parent_operation'] == 'set'):
        conflict = 'write-'
    else:
        conflict = 'read-'

    if (row['accessor_operation'] == 'set'):
        conflict += 'write'
    else:
        conflict += 'read'
    return conflict


def find_conflicts(df):
    visit_id = df[0]
    df = df[1]
    output = StringIO()
    df_conflicts = writer(output)
    df_conflicts.writerow(['visit_id', 'parent_time_stamp', 'accessor_time_stamp', 'document_url', 'cookie_key', 'parent_url', 'parent_entity',
                           'parent_value', 'accessor_url', 'accessor_entity', 'accessor_value', 'conflict_type', 'organizational_conflict'])

    unique_groups = df.groupby(['document_url', 'cookie_key'])

    for name, group in unique_groups:
        document_url = name[0]
        cookie_key = name[1]

        group.sort_values('time_stamp', ascending=True, inplace=True)

        def process_row(row):
            entity = row['script_entity']
            time_stamp = row['time_stamp']

            def add_conflicts(n_row):
                if row['operation'] == 'set':
                    conflict_type = 'write-'
                else:
                    conflict_type = 'read-'

                if n_row['operation'] == 'set':
                    conflict_type += 'write'
                else:
                    conflict_type += 'read'

                df_conflicts.writerow([visit_id, row['time_stamp'], n_row['time_stamp'], document_url, cookie_key, row['script_url'], row['script_entity'], row['cookie_value'],
                                       n_row['script_url'], n_row['script_entity'], n_row['cookie_value'], conflict_type, f'{row["party"]}-{n_row["party"]}'])

            group[(group.time_stamp.values > time_stamp) & (
                group.script_entity != entity)].apply(add_conflicts, axis=1)
        group.apply(process_row, axis=1)

    output.seek(0)
    return pd.read_csv(output, engine='python')


def parse_url_arguments(url):
    parsed = urlparse.urlparse(url)
    sep1 = parse_qs(parsed.query)
    return {**sep1}


def get_cookies_from_headers(headers):
    headers = json.loads(headers)
    for header in headers:
        if header[0] == "Cookie":
            cookies = header[1].split(";")
            cookies_dict = {}
            for cookie in cookies:
                key = cookie[0:cookie.find('=')]
                value = cookie[cookie.find('=')+1:]
                if (key in cookies_dict):
                    cookies_dict[key].append(value)
                else:
                    cookies_dict[key] = [value]
            return cookies_dict
    return {}


def compare_with_obfuscation(cookie_value, param_value, threshold):
    if (cookie_value == param_value or distance(cookie_value, param_value) < threshold):
        return True
    encoded_value = hashlib.md5(cookie_value.encode('utf-8')).hexdigest()
    if (encoded_value == param_value):
        return True

    encoded_value = hashlib.sha1(cookie_value.encode('utf-8')).hexdigest()
    if (encoded_value == param_value):
        return True

    encoded_value = base64.b64encode(
        cookie_value.encode('utf-8')).decode('utf8')
    if (encoded_value == param_value):
        return True

    return False


def check_if_cookie_value_exists(cookie_key, cookie_value, param_dict, threshold):
    cookie_value = str(cookie_value)
    if (len(cookie_value) < 5):
        return False, None

    for key in param_dict:
        if (cookie_key == param_dict[key][0]):
            return True, key
        if(len(param_dict[key][0]) < 5):
            continue
        difference = abs(len(cookie_value) - len(param_dict[key][0]))
        if difference > 10:
            continue
        offset = len(cookie_value) - difference if len(cookie_value) > len(
            param_dict[key][0]) else len(param_dict[key][0])-difference
        for i in range(0, difference+1):
            if len(param_dict[key][0]) > len(cookie_value):
                if (compare_with_obfuscation(cookie_value, param_dict[key][0][i:i+offset], threshold)):
                    return True, key
            else:
                if (compare_with_obfuscation(cookie_value[i:i+offset], param_dict[key][0], threshold)):
                    return True, key

    return False, None


def find_exfiltrations(cookies):

    output = StringIO()
    exfilrations = writer(output)
    exfilrations.writerow(['visit_id', 'exfiltration_time_stamp', 'cookie_key', 'cookie_value',
                           'exfiltrated_key', 'exfiltrated_value', 'request_url', 'request_headers'])

    def process_group(cookie_group):
        visit_id = cookie_group.visit_id.iloc[0]
        requests_for_visitid = requests[np.in1d(
            requests['visit_id'].values, [visit_id])]

        def process_cookie(cookie_row):
            cookie_key = cookie_row['cookie_key']
            cookie_value = cookie_row['cookie_value']

            def process_request(row):
                # header_cookies = get_cookies_from_headers(row['headers'])
                header_cookies = {}
                url_parameters = parse_url_arguments(row['url'])

                values_dict = {**header_cookies, **url_parameters}

                exists, key = check_if_cookie_value_exists(
                    cookie_key, cookie_value, values_dict, 2)

                if exists:
                    exfilrations.writerow([visit_id, row['time_stamp'], cookie_key, cookie_value,
                                           key, values_dict[key][0], row['url'], row['headers']])

            requests_for_visitid.apply(
                process_request, axis=1)
        cookie_group.apply(process_cookie, axis=1)
        del requests_for_visitid
    cookies.groupby(['visit_id', 'cookie_key', 'cookie_value']
                    ).apply(process_group)

    output.seek(0)
    return pd.read_csv(output, engine='python')


def find_exfiltrations_conflict(cookies):
    output = StringIO()
    exfilrations = writer(output)
    exfilrations.writerow(['visit_id', 'exfiltration_time_stamp', 'accessor_time_stamp', 'cookie_key', 'cookie_value', 'exfiltrated_key', 'exfiltrated_value',
                           'document_url', 'parent_entity', 'parent_url', 'accessor_url', 'accessor_entity', 'conflict_type', 'request_url', 'request_headers'])
    visit_id = cookies[0]
    cookies = cookies[1]
    requests_for_visitid = requests[np.in1d(
        requests['visit_id'].values, [visit_id])]

    def process_cookie(cookie_row):
        cookie_key = cookie_row['cookie_key']
        cookie_value = cookie_row['accessor_value']

        def process_request(row):
            # header_cookies = get_cookies_from_headers(row['headers'])
            header_cookies = {}
            url_parameters = parse_url_arguments(row['url'])

            values_dict = {**header_cookies, **url_parameters}

            exists, key = check_if_cookie_value_exists(
                cookie_key, cookie_value, values_dict, 1)

            if exists:
                exfilrations.writerow([visit_id, row['time_stamp'], cookie_row['accessor_time_stamp'], cookie_key, cookie_value, key, values_dict[key][0], cookie_row['document_url'],
                                       cookie_row['parent_entity'], cookie_row['parent_url'], cookie_row['accessor_url'], cookie_row['accessor_entity'], cookie_row['conflict_type'], row['url'], row['headers']])

        requests_for_visitid.apply(
            process_request, axis=1)

    cookies.apply(process_cookie, axis=1)

    output.seek(0)
    return pd.read_csv(output, engine='python')
