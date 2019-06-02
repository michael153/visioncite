"""
Formats citations into various styles (MlA, APA).
@author Michael Wan
"""

import datetime
from urllib import parse


def get_name_splice(author):
    """ Splits a name into its first, middle, and last name components.

    Arguments:
        author: The name of interest
    """
    author = author.strip()
    if " " not in author:
        return ('', '', author)
    first = author.find(" ")
    last = author.rfind(" ")
    if author.count(" ") < 2:
        return (author[:first], '', author[last + 1:])
    return (author[:first], author[first + 1:last], author[last + 1:])


def get_website_root(url):
    """ Given a url, extract the hostname.

    For instance, the hostname of the url "https://google.com/search"
    is google.com

    Arguments:
        url: The url of interest
    """
    url = url.replace("www.", "")
    if "https://" not in url and "http://" not in url:
        url = "http://" + url
    return parse.urlparse(url).netloc


class Citation:
    """ Stores citation information and provides methods for
    APA and MLA formatting
    """

    def __init__(self,
                 title="",
                 authors=None,
                 date=("", "", ""),
                 publisher="",
                 url="",
                 author_is_organization=False,
                 is_website=False):
        self.title = title
        self.authors = authors
        self.date_month = date[0]
        self.date_day = date[1]
        self.date_year = date[2]
        self.publisher = publisher
        self.url = url
        self.author_is_organization = author_is_organization
        self.is_website = is_website or url
        if self.author_is_organization:
            assert len(authors) == 1, "More than one organization as author."
        if self.is_website:
            # assert self.url is not None, "Generic website has no URL."
            if not self.url:
                print("*** Warning. Website has no URL.")

    def format_mla(self):
        """ Formats this citation class into MLA format

        For non-website works, the general format is:
            Last name, First name. <i>Title.<i> City: Publisher, Year.
        For websites, the format is:
            Contributors. "Title." Website. Edition.
            Website Publisher, Date. Web. Date Accessed.

        Source: http://www.easybib.com/reference/guide/mla/general
        """
        enquote = lambda t: "\"" + t + "\""
        italicize = lambda t: "<i>" + t + "</i>"
        components = []
        if self.authors:
            if self.author_is_organization:
                components.append(self.authors[0] + ".")
            else:
                author_components = []
                self.authors.sort(key=lambda a: get_name_splice(a)[2])
                first = get_name_splice(self.authors[0])
                author_components.append(", ".join(
                    [i for i in [first[2], first[0]] if i]))
                if len(self.authors) > 1:
                    if len(self.authors) > 2:
                        for i in range(1, len(self.authors) - 1):
                            additional = get_name_splice(self.authors[i])
                            author_components.append(" ".join([
                                i for i in [additional[0], additional[2]] if i
                            ]))
                    last = get_name_splice(self.authors[-1])
                    author_components.append(
                        "and " + " ".join([i for i in [last[0], last[2]] if i]))
                components.append(", ".join(author_components).strip() + ".")
        if self.title:
            title_str = italicize(self.title.strip().title() + ".")
            if self.is_website:
                title_str = enquote(self.title.strip().title() + ".")
            components.append(title_str)
        components.append("N.p.:")
        if self.is_website:
            components[-1] = italicize(
                get_website_root(self.url).capitalize() + ".")
        components.append("n.p.")
        if self.publisher:
            components[-1] = self.publisher.strip().title()
        is_date = self.date_year or self.date_month or self.date_day
        if not is_date:
            components[-1] += ", n.d."
        else:
            date = ""
            if self.is_website:
                date = " ".join([
                    str(d)
                    for d in [self.date_day, self.date_month, self.date_year]
                    if d
                ])
            else:
                date = str(self.date_year).strip()
            components[-1] += ", " + date + "."
        if self.is_website:
            components.append("Web.")
            components.append(datetime.datetime.now().strftime("%d %B %Y") +
                              ".")
        return " ".join(components)

    def format_apa(self):
        """ Formats this citation class into APA format

        The general APA format is:
            Last, F. M. (Date). Title. <i>Publisher.</i> Retrieved from URL.

        Source: http://www.easybib.com/reference/guide/apa/general
        """
        italicize = lambda t: "<i>" + t + "</i>"
        ret = ""
        components = []
        if self.authors:
            if self.author_is_organization:
                components.append(self.authors[0] + ".")
            else:
                self.authors.sort(key=lambda a: get_name_splice(a)[2])
                func = lambda name: name[2] + \
                       (", " + name[0][0] if name[0] else "") + "." + \
                       (" " + name[1][0] + "." if name[1] else "")
                auth_str = ", ".join([
                    func(_a)
                    for _a in map(get_name_splice,
                                  self.authors[:min(len(self.authors), 6)])
                ])
                if len(self.authors) > 6:
                    auth_str += ", et al."
                components.append(auth_str)
        is_date = self.date_year or self.date_month or self.date_day
        if is_date:
            date_str = ", ".join([str(self.date_year), str(self.date_month)])
            if self.date_month and self.date_day:
                date_str += " " + self.date_day
            date_str = "(" + date_str + ")." if date_str else date_str
            components.append(date_str)
        else:
            components.append("(n.d.).")
        if self.title:
            title_str = self.title.strip() + "."
            if self.authors:
                components.append(title_str)
            else:
                components = [title_str] + components
        if self.publisher:
            components.append(italicize(self.publisher.strip() + "."))
        if self.url:
            if not is_date:
                components.append(
                    "Retrieved " +
                    datetime.datetime.now().strftime("%B %d, %Y") + ", from " +
                    self.url + ".")
            else:
                components.append("Retrieved from " + self.url + ".")
        ret = " ".join(i.strip() for i in components if i.strip())
        return ret


TEST = Citation(title="On Global Warming and Financial Imbalances",
                authors=[
                    "John Milken", "Michael Wan", "John Test Doe",
                    "John Simerlink", "Balaji Veeramani", "Andrew Kirrilov"
                ],
                date=("May", "", 2007),
                publisher="New Perspectives Quarterly",
                url="http://test.com")

print(TEST.format_mla())
print(TEST.format_apa())
