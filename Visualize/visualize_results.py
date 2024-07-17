# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# Takes the results from a residue fingerprint match through run.py
# formatted into a csv file and plots and prints a number of
# simple statistics based on that data.

import csv
import matplotlib.pyplot as plt
import sys

def load_results(filename):
    """Reads a CSV file with PCE / SPCE results
    The lines must be formatted as:
    camera,channel_1_PCE,channel_2_PCE,channel_3_PCE
    Returns a dictionary with the camera names as keys
    and a flattened list of channel results in order as value.
    [11,12,13,21,22,23,31,32,33,41,42,43, etc]
    """
    data = {}
    with open(filename, newline='') as f:

        data_reader = csv.reader(f, delimiter=',')
        for row in data_reader:
            values = (float(row[1]), float(row[2]), float(row[3]))
            row_name = row[0].split('_')[0]
            if row_name not in data:
                data[row_name] = []
            data[row_name].extend(values)
    return data

def plot_results(data):
    """Plots a dictionary of the form described in load_results
    Includes labels based on the names, and seperates channels horizontally."""
    for i, (name, values) in enumerate(data.items()):
        x_vals = [i + 0.25 * (j % 3) for j in range(len(values))]
        plt.scatter(x_vals, values, label=name)
    plt.legend(loc='upper right')
    plt.show()

def check_threshold(data, threshold):
    """Computes the average PCE or SPCE results of each picture
    based on the data format described in load_results and
    then compares the pictures of each camera against a given
    threshold."""
    results = {}
    for name, values in data.items():
        results[name] = 0
        for i in range(len(values) // 3):
            avg_value = sum(values[3 * i + a] for a in range(3))
            if avg_value < threshold:
                results[name] += 1
    for name in results:
        print(f"Camera {name} got {results[name]/(len(data[name]) // 3)} identified correctly.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Not enough arguments, aborting.")
    else:
        data = load_results(sys.argv[1])
        plot_results(data)

        check_threshold(data, 3 * 27000)
