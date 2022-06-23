from pathlib import Path
import os

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

def download_lists(filterlist_dir: Path, overwrite: bool = False):
    
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