# Copyright 2018 Balaji Veeramani, Michael Wan
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# Author: Michael Wan <m.wan@berkeley.edu>
"""Library that provides method to standardize certain words so they can properly
located within a text"""

import re
import itertools
import datetime
import numpy as np

from dateparser.search import search_dates
from sklearn.feature_extraction.text import TfidfVectorizer

from data.storage import Table, Record
from data.queries import contains
from utils.debugging import debug
from utils.decorators import timeit


def standardize(data, datatype):
    """Define methods that standardize fields into a singular format that is logical
    and searchable
    """

    def std_markdown(markdown):
        """Remove the substring 'Image\n\n' from markdown."""
        for substring in {"Image\n\n"}:
            markdown = markdown.replace(substring, "")
        return markdown

    def std_html(html):
        """Remove script and style elements from HTML."""

        def remove_elements(html, tag_name):
            """Remove elements of a given type from HTML."""
            while "<" + tag_name in html:
                open_tag_start = html.find("<" + tag_name)
                open_tag_end = html.find(">", open_tag_start)
                close_tag_start = html.find("</" + tag_name, open_tag_end)
                close_tag_end = html.find(">", close_tag_start)
                html = html[:open_tag_start] + html[close_tag_end + 1:]
            return html

        html = remove_elements(html, "script")
        html = remove_elements(html, "style")
        return html

    def std_table(table):
        """Standardizes a default Table object so that it can be processed by
        pipeline. (i.e Author field is created from first, last, first1, last1, etc.)
        """
        std_fields = [
            "title", "author", "publisher", "date", "url", "archive-url"
        ]
        ret = Table(fields=std_fields)
        for rec in table.records:
            author_fields = [("first", "last"), ("first1", "last1"),
                             ("first2", "last2")]
            authors = []
            for i in author_fields:
                author = (rec[i[0]] + " " + rec[i[1]]).strip()
                if author != "":
                    authors.append(author)
            values = []
            for attr in std_fields:
                if contains(attr)(rec):
                    values.append(rec[attr])
                elif attr == 'author':
                    values.append(list(set(authors)))
                else:
                    values.append("")
            ret.add(Record(std_fields, values))
        return ret

    def std_text(text):
        """Standardizes a string representing article text
        Resources: https://stackoverflow.com/questions/19785458/capitalization-of-sentences-in-python
        """

        def clean_text(text):
            """Method that cleans a string to only include relevant characters and words"""
            text = text.replace('\'', '')
            text = text.replace('\"', '')
            matched_words = re.findall(r'\S+|\n', re.sub(
                "[^\w#\n]", " ", text))
            words_and_pound_newline = [
                i for i, j in itertools.zip_longest(
                    matched_words, matched_words[1:]) if i != j
            ]
            words_and_pound_newline = [('#' if '#' in x else x)
                                       for x in words_and_pound_newline]
            words_and_pound_newline = [(x.replace('_', '') if '_' in x else x)
                                       for x in words_and_pound_newline]
            ret = ''
            for i in range(len(words_and_pound_newline)):
                word = words_and_pound_newline[i]
                if i == 0 or word == '\n' or (
                        i > 0 and words_and_pound_newline[i - 1] == '\n'):
                    ret += word
                else:
                    ret += (" " + word)
            return ret

        def uppercase(matchobj):
            return matchobj.group(0).upper()

        def capitalize(s):
            return re.sub('^([a-z])|[\.|\?|\!]\s*([a-z])|\s+([a-z])(?=\.)',
                          uppercase, s)

        text = clean_text(text)
        return capitalize(text)

    def std_author(authors):
        """Method for standardizing a field if it is a list of authors"""
        return [
            ' '.join([
                name.capitalize() for name in author.replace('.', '').replace(
                    '-', ' ').split(' ')
            ]) for author in authors
        ]

    def std_date(date):
        """Method for standardizing a field if it is a date"""
        base = datetime.datetime(1000, 1, 1, 0, 0)
        matches = search_dates(
            date, settings={
                'STRICT_PARSING': True,
                'RELATIVE_BASE': base
            })
        if matches and matches[0][1] - base > datetime.timedelta(days=2 * 365):
            return matches[0][1].strftime('%m/%d/%y')
        return date.lower().replace(',', ' ').replace('-', ' ')

    def std_title(title):
        """Method for standardizing a field if it is a title"""
        return title.title()

    def std_url(url):
        """Method to standardize a url"""
        return url

    try:
        if datatype.lower() in [
                "markdown", "html", "table", "text", "author", "date", "title",
                "url"
        ]:
            return eval("std_{0}(data)".format(datatype.lower()))
        else:
            debug(
                "*** Error in standardization.standardize: {0} not standardizable"
                .format(datatype))
            return data
    except Exception as e:
        debug("*** Error in standardization.standardize: {0}".format(e))
        return ""

def find_fuzzy_fast(field, text, start=0, end=None, threshold_value=0.6):
    """Fast fuzzy string matching
    References:
    https://bergvca.github.io/2017/10/14/super-fast-string-matching.html
    http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/
    https://stackoverflow.com/questions/52048562/efficient-way-to-compute-cosine-similarity-between-1d-array-and-all-rows-in-a-2d
    https://stackoverflow.com/questions/36013295/find-best-substring-match
    """

    def generate_chunks(string, length):
        return [
            string[0 + i:length + i]
            for i in range(0, len(string), length) if length + i < len(string)
        ]

    def generate_ngrams(string, n=3):
        """Helper method for generating ngrams"""
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    try:
        text = text[start:end]
        chunks = [field] + generate_chunks(text, len(field))
        vectorizer = TfidfVectorizer(min_df=1, analyzer=generate_ngrams)
        tf_idf_matrix = vectorizer.fit_transform(chunks)
        vec, arr = tf_idf_matrix[0].toarray().flatten(
        ), tf_idf_matrix[1:].toarray()
        a, b = np.linalg.norm(arr, axis=1), np.linalg.norm(vec)
        out = (arr @ vec) / (a * b)
        index = np.argmax(out) * len(field)

        ngrams = [field]
        for i in range(
                max(0, index - len(field)), min(index + len(field),
                                                len(text))):
            if i+len(field) >= len(text):
                break
            ngrams.append(text[i:i + len(field)])

        vectorizer = TfidfVectorizer(min_df=1, analyzer=generate_ngrams)
        tf_idf_matrix = vectorizer.fit_transform(ngrams)
        vec, arr = tf_idf_matrix[0].toarray().flatten(
        ), tf_idf_matrix[1:].toarray()
        a, b = np.linalg.norm(arr, axis=1), np.linalg.norm(vec)
        out = (arr @ vec) / (a * b)
        ans = np.argmax(out) + max(0, index - len(field))
        if np.max(out) < threshold_value:
            return (-1, -1)
        ret = (ans, ans + len(field))
        return ((start + ret[0].item(), start + ret[1].item()), np.max(out))
    except Exception as e:
        return (-1, -1)


def find(field, text, datatype, start=0, end=None, threshold_value=0.6):
    """Attempts to locate a field as a substring of text based on its datatype.
    Assumes text is cleaned by pipeline's clean_text"""

    def find_generic(field, text, start=0, end=None):
        """Basic method for finding any generic field within a text"""
        text = text[start:end]
        index = text.find(field)
        if index != -1:
            return (start + index, start + index + len(field))
        return (-1, -1)

    def find_singular_author(author, text, start=0, end=None):
        """Helper method for finding a singular author"""
        return find_generic(author.lower(), text.lower(), start, end)

    def find_author(authors, text, start=0, end=None):
        """Method for finding an "author" field (a list of authors) in a text"""
        authors = standardize(authors, 'author')
        ret = []
        for author in authors:
            pos = find_singular_author(author, text, start, end)
            ret.append(pos)
        return ret

    def find_date(date, text, start=0, end=None):
        """Method for finding a date field in a text"""
        text = text[start:end]
        date = datetime.datetime.strptime(date, '%m/%d/%y')
        # Pass an impossible relative base so that relative words like "today" won't be detected
        matches = search_dates(
            text,
            settings={
                'STRICT_PARSING': True,
                'RELATIVE_BASE': datetime.datetime(1000, 1, 1, 0, 0)
            })
        if matches:
            for original_text, match in matches:
                if date.date() == match.date():
                    return find_generic(original_text.lower(), text.lower(), start, end)
        return (-1, -1)

    def find_title(title, text, start=0, end=None):
        """Method for finding a title field in a text
        ?? 'http://www.digitalspy.com/gaming/retro-corner/news/a381156/retro-corner-wolfenstein-3d/'
        """
        # return find_generic(title.lower(), text.lower())
        return find_fuzzy_fast(title.lower(), text.lower(), start, end, threshold_value)[0]

    try:
        if datatype.lower() in ["author", "date", "title"]:
            return eval("find_{0}(field, text)".format(datatype.lower()))
        else:
            debug(
                "*** Warning in standardization.find: {0} not a findable field"
                .format(datatype))
            return find_generic(field, text, start, end)
    except Exception as e:
        debug("*** Error in standardization.find: {0}".format(e))
        return (-1, -1)


def clean_to_ascii(foreign_char):
    """Converts a non-ASCII character into it's ASCII equivalent

        >>> clean_to_ascii('ç')
        'c'
    """
    special_chars = {
        'a': ['à', 'á', 'â', 'ä', 'æ', 'ã', 'å', 'ā'],
        'c': ['ç', 'ć', 'č'],
        'e': ['è', 'é', 'ê', 'ë', 'ē', 'ė', 'ę'],
        'i': ['î', 'ï', 'í', 'ī', 'į', 'ì'],
        'l': ['ł'],
        'n': ['ñ', 'ń'],
        'o': ['ô', 'ö', 'ò', 'ó', 'œ', 'ø', 'ō', 'õ'],
        's': ['ß', 'ś', 'š'],
        'u': ['û', 'ü', 'ù', 'ú', 'ū'],
        'y': ['ÿ'],
        'z': ['ž', 'ź', 'ż'],
        'A': ['À', 'Á', 'Â', 'Ä', 'Æ', 'Ã', 'Å', 'Ā'],
        'C': ['Ç', 'Ć', 'Č'],
        'E': ['È', 'É', 'Ê', 'Ë', 'Ē', 'Ė', 'Ę'],
        'I': ['Î', 'Ï', 'Í', 'Ī', 'Į', 'Ì'],
        'L': ['Ł'],
        'N': ['Ñ', 'Ń'],
        'O': ['Ô', 'Ö', 'Ò', 'Ó', 'Œ', 'Ø', 'Ō', 'Õ'],
        'S': ['Ś', 'Š'],
        'U': ['Û', 'Ü', 'Ù', 'Ú', 'Ū'],
        'Y': ['Ÿ'],
        'Z': ['Ž', 'Ź', 'Ż']
    }
    if foreign_char in sum(special_chars.values(), []):
        for k in special_chars:
            if foreign_char in special_chars[k]:
                return k
    debug("Can't convert: " + str(foreign_char))
    return ' '
