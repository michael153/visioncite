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

import datetime
import inspect
import io
import json
import os
import os.path
import re
import string
import sys
import time
import requests

import assets
import data.standardization as standardization
import data.queries as queries
from data.storage import Table
from utils.decorators import timeout
from utils.debugging import debug

import findBoundedBox
from pipeline import get_image_mappings

def get_wiki_article_links_info(file, args, num=False, already_collected=[]):
    """Retrieve article information from wikipedia database Tables, and store
    data into a tupled list
    >>> get_wiki_article_links_info('asserts/data.txt', ['url', 'author'])
    """
    debug("Reading Wikipedia Article Links from...", file)
    start_time = time.time()
    table = standardization.standardize(Table(file), 'Table').query(
        queries.contains(*args))
    data = []
    total = 0
    for rec in table.records:
        url = rec['url']
        if url not in already_collected:
	        data.append(tuple([rec[a] for a in args]))
	        total += 1
        if num and total == num:
            break

    # data = [tuple([rec[a] for a in args]) for rec in table.records]
    # Return labels in order to remember what each index in a datapoint represents
    labels = {args[x]: x for x in range(len(args))}
    debug("Links successfully collected in {0} seconds\n".format(time.time() -
                                                                 start_time))
    return (data, labels)


def collect_dict(info):
    """Collect info and manipulate into the proper format to be saved
    as data
    Arguments:
        info, a tuple containing data points, and a label lookup dict
    """

    datapoints, label_lookup = info[0], info[1]
    data = {}
    debug("Getting {0} points...".format(len(datapoints)))
    for entry in datapoints:
        url = entry[label_lookup['url']]
        citation_dict = {
            x: entry[label_lookup[x]]
            for x in label_lookup.keys()
        }
        try:
            entry = {
                'url': url,
                'citation_info': citation_dict,
            }
            data[url] = entry
        except Exception as e:
            func_name = inspect.getframeinfo(inspect.currentframe()).function
            print(colored(">>> Error in {0}: {1}".format(func_name, e), "red"))
    return data

def get_saved_keys(file_name):
    """Given a file_name, collect the saved data and return a data dict"""
    if not os.path.isfile(file_name):
        print(colored(">>> Error: Opening file {0}".format(file_name), "red"))
        return []
    saved_dict = json.load(open(file_name))
    return list(saved_dict.keys())

def save_dict(data):
	with open(assets.DATA_PATH + '/data_dict.json', 'w') as out:
		json.dump(data, out, sort_keys=True, indent=4)

def supervised_find_title(url, blobs):
	info = DATA_DICT[url]
	title = info['citation_info']['title']
	print(title + ": " + url)
	try:
		for blob in blobs:
			text = blob[1]
			print(text)
			loc, threshold = standardization.find_fuzzy_fast(title, text, threshold_value=0.0)
			if threshold > 0.7:
				return (blob, loc)
	except:
		return None

ALREADY_COLLECTED_KEYS = get_saved_keys(assets.DATA_DICT + '/image_mappings.json')

WIKI_FILE_PATH = assets.DATA_PATH + '/citations.csv'
INFO = get_wiki_article_links_info(WIKI_FILE_PATH,
	['url', 'title', 'author', 'date'],
	already_collected=ALREADY_COLLECTED_KEYS)
DATA_DICT = collect_dict(INFO)

if __name__ == '__main__':
	url = "http://allhiphop.com/2015/01/19/yukmouth-2pac-impact-notorious-big-versace/"
    image_mappings = get_image_mappings_json()
	k = supervised_find_title(url, findBoundedBox.get_blobs_by_url(url, image_mappings[url], display=True, useSavedData=True))
	print("\n\n")
	print(k)