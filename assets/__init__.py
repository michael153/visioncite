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
import os

PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = PATH + "/data"
MODELS_PATH = PATH + "/data/ml"
TRAINING_PATH = DATA_PATH + "/training"

SEGMENTS_PATH = PATH + "/segments"
IMAGE_PATH = DATA_PATH + "/images"
XML_PATH = DATA_PATH + "/xml"
JSON_PATH = DATA_PATH + "/json"
