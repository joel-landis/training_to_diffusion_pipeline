import csv
import json
import os


def write_text(path, text):
    """Write text to path and fsync it."""
    with open(path, "w") as output_file:
        output_file.write(text)
        output_file.flush()
        os.fsync(output_file.fileno())


def write_json(path, data):
    """Write data as indented JSON to path and fsync it."""
    with open(path, "w") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.flush()
        os.fsync(output_file.fileno())


def write_csv(path, header, rows):
    """Write header and rows to a CSV file and fsync it."""
    with open(path, "w", newline="") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(header)
        csv_writer.writerows(rows)
        output_file.flush()
        os.fsync(output_file.fileno())
