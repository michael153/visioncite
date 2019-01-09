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
# Author: Balaji Veeramani <bveeramani@berkeley.edu>
"""Define query functions."""


def contains(*fields):
    """Return true if record contains values for all given fields."""

    def query(record):
        for field in fields:
            if not field in record:
                return False
        return True

    return query


def either(query1, query2):
    """Return true if record satisfies either query1 or query2."""

    def query(record):
        return query1(record) or query2(record)

    return query


def both(query1, query2):
    """Return true if record satisfies both query1 and query2."""

    def query(record):
        return query1(record) and query2(record)

    return query


def negate(query1):
    """Return true if record does NOT satisfy query1."""

    def query(record):
        return not query1(record)

    return query
