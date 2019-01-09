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
"""Define function and objects for manipulating data files."""
from utils import multithreading


def csv(item, delimiter):
    """Return the csv-valid representation of an object."""
    return type(item).__csv__(item, delimiter)


class Table:
    """A generic data table."""

    DELIMITER = "\t"

    def __init__(self, filename=None, fields=()):
        """Initialize a data table.

        Arguments:
            fields: A list of attribute names.
            records: A list of Record objects.
            filename: The name of a data sheet.
        """
        self.dictionary = {}
        if fields and not hasattr(fields, "__iter__"):
            raise TypeError("fields must be a collection.")
        self.fields = fields
        if filename:
            self.load(filename)

    def load(self, filename, num_records=1000000):
        """Load data from a file and return table.

        By default, the load method will add at most one million records.

        Arguments:
            filename: The name of some data file
            num_records: How many records should be loaded
        """
        with open(filename, encoding="utf-8") as file:
            lines = file.read().splitlines()
        if not self.fields:
            self.fields = lines[0].split(self.DELIMITER)
        for line in lines[1:num_records + 1]:
            self.add(self.parse(line))
        return self

    def parse(self, line):
        """Parse a line of text that represents a data record."""
        values = line.split(self.DELIMITER)
        return Record(self.fields, values)

    def save(self, filename):
        """Save data to a file."""
        with open(filename, "w", encoding="utf-8") as file:
            file.write(self.header + "\n")
            for record in self.records:
                file.write(csv(record, self.DELIMITER) + "\n")

    @property
    def header(self):
        """A string representing this table's header."""
        header = ""
        for field in self.fields:
            header += field + self.DELIMITER
        return header.rstrip()

    @property
    def records(self):
        """Return a list of this table's records."""
        return list(self.dictionary.values())

    def find(self, field, value):
        """Return the first record that contains the given field-value pair.

        If no such values are found, the method returns None.

        Arguments:
            field: A record field
            value: A record value
        """
        for record in self.records:
            if record[field] == value:
                return record
        return None

    def query(self, function):
        """Return a Table containing records that satisfy some function."""
        valid = []

        def validate(*records):
            for record in records:
                if function(record):
                    valid.append(record)

        # Run the query with six seperate threads
        threads = multithreading.build(6, validate, self.records)
        multithreading.execute(threads)

        result = Table(fields=self.fields)
        for record in valid:
            result.add(record)
        return result

    def add(self, record, key=None):
        """Add record to the end of this table.

        The fields of the records must match the fields of the table.

        Arguments:
            record: A Record object with the same fields as this table
            key: The desired key for the record (optional)
        """
        if not hasattr(record, 'fields') or not hasattr(record, 'values'):
            raise TypeError("Expected object with fields and values.")
        if record.fields != self.fields:
            raise ValueError("Table and record fields are mismatched.")
        if key in self.dictionary:
            raise ValueError("A record with that key already exists.")
        key = key or len(self.records)
        self.dictionary[key] = record

    def extend(self, records):
        """Add collection of records to the table.

        Arguments:
            records: A collection of Record objects.
        """
        if not hasattr(records, '__iter__'):
            raise TypeError("records must be iterable.")
        for record in records:
            self.add(record)

    def __contains__(self, record):
        for value in self.dictionary.values():
            if record == value:
                return True
        return False

    def __getitem__(self, key):
        if key not in self.dictionary:
            raise KeyError("Table has no record with key " + str(key))
        return self.dictionary[key]

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __eq__(self, other):
        if not isinstance(other, Table):
            return False
        return self.dictionary == other.dictionary


class Record:
    """A record in a data table."""

    LABEL_SIZE = 5

    def __init__(self, fields, values):
        """Initialize a record.

        Arguments:
            fields: A list representing the record's attributes.
            values: A list representing the value of each attribute.
        """
        self.fields, self.values = fields, values

    @property
    def data(self):
        """A dictionary representing a record's data."""
        return dict(zip(self.fields, self.values))

    def __getitem__(self, field):
        return self.data.get(field, "")

    def __contains__(self, field):
        return bool(self.data.get(field, ""))

    def __eq__(self, other):
        if not isinstance(other, Record):
            return False
        if not self.fields == other.fields:
            return False
        return self.values == other.values

    def __csv__(self, delimiter):
        """Return csv-compatible representation."""
        string = ""
        for field in self.data:
            string += self.data[field] + delimiter
        return string.rstrip()

    def __repr__(self):
        return "Record({0}, {1})".format(self.fields, self.values)

    def __str__(self):
        string = ""
        for field in self.data:
            if self.data[field]:
                string += field[:self.LABEL_SIZE] + "\t"
                string += str(self.data[field]) + "\n"
        return string.rstrip()
