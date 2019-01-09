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

# import sys, time, string, random
# import numpy as np
# from difflib import SequenceMatcher

# from core.pipeline import slice_text
# from data.storage import Table
# from web.webpages import Webpage
# from utils.debugging import debug
# from data.queries import contains

# import assets
# import core.pipeline as pipeline
# import data.standardization as standardization

# def accuracy_fuzzy_match(sample, min_length=10, max_edits=3, strings_per_url=100):
#     """Return the accuracy of the fuzzy_match algorithm in standardization.find
#     """
#     debug("Collecting seed urls...")
#     contents = []
#     edit_types = ["swap", "add", "del"]
#     errors_by_edit_type = {e: {"error": [], "total": 0} for e in edit_types}
#     errors_by_num_edits = {e: {"error": [], "total": 0} for e in range(max_edits+1)}
#     char_dict = string.ascii_uppercase + string.ascii_lowercase + string.digits + "\n_-#"
#     success, total, wrong_string_found, no_string_found = 0, 0, [], []
#     for record in sample:
#         try:
#             contents.append(pipeline.get_content_from_url(record["url"]))
#         except Exception as ex:
#             debug("Error: {0} | {1}".format(ex, record["url"]))
#     for content in contents:
#         if len(content) < 2:
#             continue
#         for num_edits in range(max_edits+1):
#             for i in range(strings_per_url):
#                 s, e = random.sample(range(0, len(content)), 2)
#                 s, e = min(s, e), max(s, e)
#                 e = max(e, s + min_length)
#                 llen = len(content[s:e]) - len(content[s:e].lstrip())
#                 rlen = len(content[s:e]) - len(content[s:e].rstrip())
#                 s += llen
#                 e -= rlen
#                 if e <= s:
#                     continue
#                 field = content[s:e]
#                 edits = []
#                 for _ in range(num_edits):
#                     edit_type = np.random.randint(len(edit_types))
#                     edits.append(edit_type)
#                     if len(field) == 0:
#                         break
#                     incident_index = np.random.randint(len(field))
#                     if edit_type == 0:
#                         field = field[:incident_index] + random.choice(char_dict) + field[incident_index+1:]
#                     elif edit_type == 1:
#                         field = field[:incident_index] + random.choice(char_dict) + field[incident_index:]
#                     elif edit_type == 2:
#                         field = field[:incident_index] + field[incident_index+1:]
#                     errors_by_edit_type[edit_types[edit_type]]["total"] += 1
#                 loc = standardization.find(field, content, "title", threshold_value=0.5)
#                 if loc == (-1, -1) or abs(loc[0] - s) >= max_edits or abs(loc[1] - e) >= max_edits:
#                     for e in edits:
#                         errors_by_edit_type[edit_types[e]]["error"].append((field, record["url"]))
#                     if loc != (-1, -1):
#                         debug("Original: ({0}, {1}) \t| Found: ({2}, {3}) \t| Wrong String Found \t| {4} edits | {5}".format(s, e, loc[0], loc[1], num_edits, record["url"]))
#                         wrong_string_found.append(num_edits)
#                     else:
#                         debug("Original: ({0}, {1}) \t| Found: ({2}, {3}) \t| String Not Found \t\t| {4} edits | {5}".format(s, e, loc[0], loc[1], num_edits, record["url"]))
#                         no_string_found.append(num_edits)
#                     errors_by_num_edits[num_edits]["error"].append((field, record["url"]))
#                 else:
#                     success += 1
#                     debug("Original: ({0}, {1}) \t| Found: ({2}, {3}) \t| String Found \t\t| {4} edits | {5}".format(s, e, loc[0], loc[1], num_edits, record["url"]))
#                 errors_by_num_edits[num_edits]["total"] += 1
#                 total += 1
#     debug("Test complete.\n\nSUMMARY\n-------")
#     debug("Accuracy: {0}/{1} = {2:.2f}%\n".format(success, total, 100.0*success/total))
#     debug("Wrong String Found: {0}".format(len(wrong_string_found)))
#     debug("No String Found: {0}".format(len(no_string_found)))

#     debug("\nError breakdown by edit type:")
#     for error in errors_by_edit_type:
#         e, t = len(errors_by_edit_type[error]["error"]), errors_by_edit_type[error]["total"]
#         debug("Edit type '{0}': {1}/{2} ({3:.2f}% accuracy)".format(error, e, t, 100.0*(1 - e/t)))
#     debug("\nError breakdown by number of edits:")
#     for num in range(max_edits+1):
#         e, t = len(errors_by_num_edits[num]["error"]), errors_by_num_edits[num]["total"]
#         debug("{0} edits: {1}/{2} ({3:.2f}% accuracy)".format(num, e, t, 100.0*(1 - e/t)))
#     debug("\nNo_edit String Errors\n-------")
#     for no_edit_strings in errors_by_num_edits[0]["error"]:
#         debug("\"{0}\" | {1}\n".format(no_edit_strings[0] if len(no_edit_strings[0]) < 100 else no_edit_strings[0][:97] + "...", no_edit_strings[1]))


# def accuracy_data_preservation(sample,
#                                considered_fields=['title', 'author', 'date']):
#     """Return the average correctness of webpage content against a sample.

#     Accuracy is defined as the number of values found in the the content
#     divided by the number of values expected to be found in the content. We
#     expect the webpage content to contain all of the values that are defined in
#     the spreadsheet that are also found in the source code.

#     Arguments:
#         sample: A Table instance of sample data
#     """
#     debug("Running accuracy statistic for data_preservation...")
#     debug("Testing fields: ", considered_fields)
#     accuracies = []
#     accuracy_breakdown = {
#         x: {
#             "found": 0,
#             "total": 0
#         }
#         for x in considered_fields
#     }
#     no_title_found, urls_with_errors = [], []
#     skips = 0
#     for record in sample:
#         try:
#             if ".pdf" in record["url"]:
#                 skips += 1
#                 continue
#             start_time = time.time()
#             defined_values = [(record[field], field)
#                               for field in considered_fields
#                               if (record[field])]
#             content = Webpage(record["url"]).content
#             standardized_content = slice_text(
#                 standardization.standardize(content, "text"))
#             values_found_in_content, values_expected_in_content = 0, 0
#             for value, field in defined_values:
#                 location = standardization.find(value, standardized_content,
#                                                 field)
#                 if isinstance(location, list):
#                     for loc in location:
#                         values_expected_in_content += 1
#                         accuracy_breakdown[field]['total'] += 1
#                         if loc != (-1, -1):
#                             accuracy_breakdown[field]['found'] += 1
#                             values_found_in_content += 1
#                 else:
#                     values_expected_in_content += 1
#                     accuracy_breakdown[field]['total'] += 1
#                     if location != (-1, -1):
#                         accuracy_breakdown[field]['found'] += 1
#                         values_found_in_content += 1
#                     elif field == 'title':
#                         no_title_found.append(record["url"])

#             accuracy = values_found_in_content / values_expected_in_content if values_found_in_content else 0
#             debug("({0} : {1}) | {2} | {3:.2f}s".format(
#                 values_found_in_content, values_expected_in_content,
#                 record["url"],
#                 time.time() - start_time))
#             accuracies.append(accuracy)
#         except Exception as ex:  #pylint: disable=broad-except
#             debug("Error   | {0} | {1}".format(record["url"], ex))
#             urls_with_errors.append(record["url"])
#     average_accuracy = np.average(accuracies)
#     debug("Test complete.\n\nSUMMARY\n-------")
#     debug("Average Accuracy: {0}".format(average_accuracy))
#     for field in accuracy_breakdown:
#         debug("Field '{0}': ({1} / {2} found)".format(
#             field, accuracy_breakdown[field]['found'],
#             accuracy_breakdown[field]['total']))
#     debug("\n{0} Records Tested".format(len(sample)))
#     debug("{0} Records Skipped (is .pdf)".format(skips))
#     debug("{0} Records With Errors".format(len(urls_with_errors)))
#     debug("\n\n\nUrls w/ no title found: ")
#     debug(no_title_found)
#     debug("\nUrls w/ errors: ")
#     debug(urls_with_errors)
#     return average_accuracy


# def accuracy_title_content_extractor(sample):
#     """Return average accuracy at determining where content starts.

#     This function compares predicted webpage titles, as they are defined by
#     Webpage.content, against the actual webpage titles.

#     Arguments:
#         sample: A Table instance of sample data
#     """
#     debug("Running accuracy statistic for title_content_extractor...")
#     no_title_found, locs, urls_with_errors = [], [], []
#     skips = 0
#     for record in sample:
#         try:
#             start_time = time.time()
#             webpage = Webpage(record["url"])
#             expected_title = record["title"]
#             if ".pdf" in record["url"] or expected_title == "":
#                 skips += 1
#                 continue
#             content = Webpage(record["url"]).content
#             standardized_content = slice_text(
#                 standardization.standardize(content, "text"))
#             loc = standardization.find(expected_title, standardized_content,
#                                        "title", threshold_value=0.6)
#             if loc == (-1, -1):
#                 no_title_found.append(record["url"])
#                 debug("Title not found | {1} | {2:.2f}s".format(
#                     loc, record["url"],
#                     time.time() - start_time))
#             else:
#                 locs.append(loc[0])
#                 found_title_score = SequenceMatcher(
#                     None, standardized_content[loc[0]:loc[1]].lower(),
#                     expected_title.lower()).ratio()
#                 debug(
#                     "Title located at index {0} | {1} | {2} | {3:.2f}s".format(
#                         loc[0], found_title_score, record["url"],
#                         time.time() - start_time))
#         except Exception as ex:
#             debug("Error   | {0} | {1}".format(record["url"], ex))
#             urls_with_errors.append(record["url"])
#     debug("Test complete.\n\nSUMMARY\n-------")
#     debug("Average location of title: {0}".format(np.average(locs)))
#     debug("Median location of title: {0}".format(np.median(locs)))
#     debug("Stddev of title location: {0}".format(np.std(locs)))
#     hist = np.histogram(locs, bins=10)
#     for i in range(0, len(hist[1]) - 1):
#         debug("Number of titles ∈ [{0}, {1}): {2}".format(
#             hist[1][i], hist[1][i + 1], hist[0][i]))
#     largest_bucket_id = np.argmax(hist[0])
#     largest_bucket = [
#         x for x in locs if x >= hist[1][largest_bucket_id]
#         and x < hist[1][largest_bucket_id + 1]
#     ]
#     debug("\n")
#     largest_hist = np.histogram(largest_bucket, bins=4)
#     for i in range(0, len(largest_hist[1]) - 1):
#         debug("Number of titles ∈ [{0}, {1}): {2}".format(
#             largest_hist[1][i], largest_hist[1][i + 1], largest_hist[0][i]))

#     debug("\n{0} Records in Sample".format(len(sample)))
#     debug("{0} Records Skipped (is .pdf)".format(skips))
#     debug("{0} Records With Errors".format(len(urls_with_errors)))
#     success = len(sample) - (skips + len(no_title_found) + len(urls_with_errors))
#     total = len(sample) - (skips + len(urls_with_errors))
#     debug("{0}/{1} ({2:.2f}%) titles found".format(success, total, 100.0*success/total))
#     debug("\n\n\nUrls w/ no title found: ")
#     debug(no_title_found)
#     debug("\nUrls w/ errors: ")
#     debug(urls_with_errors)


# if __name__ == '__main__':
#     methods = ["accuracy_data_preservation", "accuracy_title_content_extractor", "accuracy_fuzzy_match"]
#     default_num_points = {
#         "accuracy_data_preservation": 150,
#         "accuracy_title_content_extractor": 150,
#         "accuracy_fuzzy_match": 15
#     }
#     if len(sys.argv) > 1:
#         if sys.argv[1] in methods:
#             file_name = "citations.csv"
#             debug("Gathering points from {0}...".format(file_name))
#             table = Table(assets.DATA_PATH + "/{0}".format(file_name))
#             num_points = default_num_points[sys.argv[1]]
#             if len(sys.argv) > 2 and sys.argv[2].isnumeric():
#                 num_points = int(sys.argv[2])
#             random_records = random.sample(table.records, num_points)
#             new_table = Table(fields=table.fields)
#             new_table.extend(random_records)
#             new_table = new_table.query(contains("title"))
#             new_table = standardization.standardize(new_table, 'table')
#             eval("{0}({1})".format(sys.argv[1], "new_table"))

