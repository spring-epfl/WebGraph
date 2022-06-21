from random import randint
import random
import string
import tldextract
from six.moves.urllib.parse import urlparse, parse_qs, parse_qsl

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def id_generator(size=6, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def randomize_qs_val(complete_url):
    keyword_raw = ["ad", "ads", "advert", "popup", "banner", "sponsor", "iframe", "googlead", "adsys", "adser", "advertise", "redirect", 
                 "popunder", "punder", "popout", "click", "track", "play", "pop", "prebid", "bid", "pb.min", "affiliate", "ban",  
                 "delivery", "promo","tag", "zoneid", "siteid", "pageid", "size", "viewid", "zone_id", "google_afc" , "google_afs"]
    screen_resolution = ["screenheight", "screenwidth", "browserheight", "browserwidth", "screendensity", "screen_res", "screen_param", "screenresolution", "browsertimeoffset"]
    keyword_raw += screen_resolution

    query_string = parse_qsl(complete_url)
    updated_url = ''
    if query_string:
        for item in query_string:
            rand_or_not = randint(1, 2)
            #Addition check
            for kw in keyword_raw:
                if kw in item[0]:
                    rand_or_not = 1
                    break
            if rand_or_not == 1:
                number_check = item[1].isnumeric()
                if number_check:
                    replacement = random_with_N_digits(len(item[1]))
                else:
                    replacement = id_generator(size = len(item[1]))
            else:
                replacement = item[1]
            updated_url += item[0] + '=' + str(replacement) + '&'
    else:
        updated_url = complete_url
    if updated_url.endswith('&'):
        updated_url = updated_url[:-1]
    return updated_url

def randomize_qs_param(complete_url):

    keyword_raw = ["ad", "ads", "advert", "popup", "banner", "sponsor", "iframe", "googlead", "adsys", "adser", "advertise", "redirect", 
                 "popunder", "punder", "popout", "click", "track", "play", "pop", "prebid", "bid", "pb.min", "affiliate", "ban",  
                 "delivery", "promo","tag", "zoneid", "siteid", "pageid", "size", "viewid", "zone_id", "google_afc" , "google_afs"]
    screen_resolution = ["screenheight", "screenwidth", "browserheight", "browserwidth", "screendensity", "screen_res", "screen_param", "screenresolution", "browsertimeoffset"]
    keyword_raw += screen_resolution

    query_string = parse_qsl(complete_url)
    updated_url = ''
    updated_url = query_string[0][0] + '=' + query_string[0][1]
    if query_string:
        for item in query_string[1:]:
            rand_or_not = randint(1, 2)
            #Addition check
            for kw in keyword_raw:
                if kw in item[0]:
                    rand_or_not = 1
                    break
            if rand_or_not == 1:
                replacement = id_generator(size=len(item[0]))
            else:
                replacement = item[0]
            updated_url += str(replacement) + '=' + item[1] + '&'
    else:
        updated_url = complete_url
    if updated_url.endswith('&'):
        updated_url = updated_url[:-1]
    return updated_url

def randomize_qs_size(complete_url):
    query_string = parse_qsl(complete_url)
    updated_url = query_string[0][0] + '=' + query_string[0][1]
    replacement = '&'
    rand_or_not = randint(1, 2)
    min_additions = 1
    max_additions = 8 #4
    max_size = 16 #8
    min_size = 1
    if query_string:
        if rand_or_not == 1:  # 1 for add 2 for delete
            number_of_additions = randint(min_additions, max_additions)
            for i in range(0, number_of_additions):
                replacement += id_generator(size=randint(min_size, max_size)) + '=' + id_generator(size=randint(min_size, max_size)) + '&'
            updated_url += replacement
        else:
            number_of_removals = randint(0, len(query_string) - 1)
            index_to_ignore = random.sample(range(1, len(query_string)), number_of_removals)
            for i in range(1, len(query_string)):
                if i not in index_to_ignore:
                    updated_url += query_string[i][0] + '=' + query_string[i][1] + '&'
    if updated_url.endswith('&'):
        updated_url = updated_url[:-1]
    return updated_url

def randomize_domain_name(complete_url):
    size_increment = 16#8
    element_domain = tldextract.extract(complete_url).domain
    return complete_url.replace(element_domain, id_generator(size=randint(len(element_domain), len(element_domain)+size_increment)), 1)

def randomize_subdomain_name(complete_url):
    max_size = 16#8
    element_subdomain = tldextract.extract(complete_url).subdomain
    element_domain = tldextract.extract(complete_url).domain
    if element_subdomain.strip() == '':
        return complete_url.replace(element_domain, id_generator(size=randint(0, max_size)) + '.' + element_domain, 1)
    else:
        return complete_url.replace(element_subdomain, id_generator(size=randint(len(element_subdomain), len(element_subdomain)+8)), 1)

def create_first_party_subdomain(top_level_url):

    max_size = 16
    element_domain = tldextract.extract(top_level_url).domain
    new_subdomain = id_generator(size=randint(0, max_size)) + '.' + element_domain
    return top_level_url.replace(element_domain, new_subdomain, 1)

def obfuscate(complete_url, party_value, top_level_url=""):

    try:
        query_string = parse_qsl(complete_url)
    except Exception as e:
        print(e)
        query_string = None
    
    randomize_querystring = randint(1, 4)
    randomize_domain = randint(1, 3)
    #randomize_both = randint(1, 3)
    randomize_both = 3
    
    if randomize_both == 1:
        if query_string:
            if randomize_querystring == 1:
                complete_url = randomize_qs_val(complete_url)
            elif randomize_querystring == 2:
                complete_url = randomize_qs_param(complete_url)
            elif randomize_querystring == 3:
                complete_url = randomize_qs_size(complete_url)
            elif randomize_querystring == 4:
                complete_url = randomize_qs_size(randomize_qs_param(randomize_qs_val(complete_url)))

    elif randomize_both == 2 and party_value == 3:
        if randomize_domain == 1:
            complete_url = randomize_domain_name(complete_url)
            # if sub_domain_value == 1:
            #     sub_domain_value = 0
        elif randomize_domain == 2:
            complete_url = randomize_subdomain_name(complete_url)
            # if party_value == 1:
            #     sub_domain_value = 1
        elif randomize_domain == 3:
            complete_url = randomize_subdomain_name(randomize_domain_name(complete_url))
            # party_value = 3
            # sub_domain_value = 0

    elif randomize_both == 3:
        if query_string:
            if randomize_querystring == 1:
                complete_url = randomize_qs_val(complete_url)
            elif randomize_querystring == 2:
                complete_url = randomize_qs_param(complete_url)
            elif randomize_querystring == 3:
                complete_url = randomize_qs_size(complete_url)
            elif randomize_querystring == 4:
                complete_url = randomize_qs_size(randomize_qs_param(randomize_qs_val(complete_url)))
        if party_value == 3:
            if randomize_domain == 1:
                complete_url = randomize_domain_name(complete_url)
                # party_value = 3
                # if sub_domain_value == 1:
                #     sub_domain_value = 0
            elif randomize_domain == 2:
                complete_url = randomize_subdomain_name(complete_url)
                # if party_value == 1:
                #     sub_domain_value = 1
            elif randomize_domain == 3:
                complete_url = randomize_subdomain_name(randomize_domain_name(complete_url))
                # party_value = 3
                # sub_domain_value = 0
        elif party_value == 1:
            complete_url = create_first_party_subdomain(top_level_url)
            # sub_domain_value = 1

    return complete_url

