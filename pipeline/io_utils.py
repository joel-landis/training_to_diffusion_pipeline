import csv
import json
import os


def split_name(path):
    """Split name of a file: the text after the last "_" of the file name without extension (selection_train.traj -> train)."""
    return os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1]


def unique_split_names(split_files):
    """Split name of each file, in input order; raises if two files give the same split name."""
    split_names = [split_name(split_file) for split_file in split_files]
    if len(set(split_names)) != len(split_names):
        raise ValueError(
            f"frame files {split_files} give non-unique split names {split_names}")
    return split_names


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
