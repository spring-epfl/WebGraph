import tldextract


def get_single_domain(url):
    u = tldextract.extract(url)
    return u.domain + "." + u.suffix


def get_domain(urls):
    try:
        if (isinstance(urls, list)):
            domains = []
            for url in urls:
                domains.append(get_single_domain(url))
            return domains
        return get_single_domain(urls)
    except:
        return None
