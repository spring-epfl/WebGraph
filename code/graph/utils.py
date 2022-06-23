import tldextract
import numpy as np
import pandas as pd
from typing import Union

def get_cookiedom_key(name, domain):

    """Function to get cookie key, a combination of the cookie name and domain.

    Args:
        name: cookie name
        domain: cookie domain
    Returns:
        cookie key
    """

    try:
        return name + '|$$|' + domain
    except:
        return name
    return name

def get_domain(url:Union[str, list[str]]) -> Union[str, list[str]]:

    """Function to get eTLD+1 from a URL.

    Args:
        url: URL
    Returns:
        eTLD+1
    """

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

def get_original_cookie_setters(df):

    """Function to get the first setter of a cookie.

    Args:
        df: DataFrame representation of all cookie sets.
    Returns:
        DataFrame representation of the cookie setter.
    """

    df_owners = {}
    df.sort_values('time_stamp', ascending=False, inplace=True)
    grouped = df.groupby(['visit_id', 'dst'])
    rows_added = 0

    for name, group in grouped: 
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
