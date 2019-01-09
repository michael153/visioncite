# Copyright 2018 Michael Wan
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

import time
import json
import random
import string
import subprocess
from subprocess import DEVNULL, STDOUT
from urllib import request
from urllib.request import Request
from fake_useragent import UserAgent
import imgkit

import images
import assets
import pipeline

def gowitness_capture(url):
    start = time.time()
    subprocess.call("nohup gowitness single -u {0} -R '800,1200' -d {1} 2&>1 &".format(url, images.RAW_IMAGES_PATH), shell=True, stderr=DEVNULL, stdout=DEVNULL)
    print("Time for gowitness image capture: {0:.2f}".format(time.time() - start))

def imgkit_capture(url, filename):
    options = {
        'width': 1000,
        'crop-w': 1000,
        'height': 1200,
        'disable-javascript': None,
        'stop-slow-scripts': None,
        # 'no-images': None,
    }
    start = time.time()
    try:
        imgkit.from_url(url, images.RAW_IMAGES_PATH + '/{0}.jpg'.format(filename), options)
        print("Time for imgkit image capture: {0:.2f}".format(time.time() - start))
        return True
    except Exception as e:
        print("Error in imgkit image capture: {0}".format(e))
        return False

# urls = []
# urls.append("https://www.cnn.com/2018/10/12/middleeast/khashoggi-saudi-turkey-recordings-intl/index.html")
# urls.append("https://www.nytimes.com/2018/09/25/us/politics/deborah-ramirez-brett-kavanaugh-allegations.html")
# urls.append("http://rogerebert.suntimes.com/apps/pbcs.dll/article?AID=/20010615/REVIEWS/106150301/1023")
# urls.append("http://cdnedge.bbc.co.uk/1/hi/england/manchester/6529795.stm")
# urls.append("http://www.latimes.com/entertainment/movies/moviesnow/la-et-mn-runner-runner-20131004-story.html")
# urls.append("http://www.flightglobal.com/news/articles/circuit-breaker-at-heart-of-lot-767-gear-up-landing-probe-365584/")

# gowitness_capture(url)
counter, threshold = 0, 700

keys = list(pipeline.data_dict.keys())
random.shuffle(keys)
for website in keys:
    if website not in pipeline.image_dict:
        print("Capturing {0}...".format(website))
        counter += 1
        if counter == threshold:
            break
        filename = str(int(time.time())) + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        url = pipeline.data_dict[website]["url"]
        out = imgkit_capture(url, filename)
        if out:
            pipeline.image_dict[url] = filename
    else:
        print("Website {0} already captured...".format(website))

pipeline.save_image_mappings(pipeline.image_dict)




# for url in urls:
    # imgkit_capture(url)


