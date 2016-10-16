"""
Creating the vocabulary.

Terms:
    mni -- mass and intensity.
"""

import math
import pymzml
import pandas as pd
import numpy as np
np.set_printoptions(precision=3, threshold=np.inf)

INTENSITY_FLOOR = 10
TOLERANCE_FACTOR = 7
DATA_PATH = "../data/"
DATA_FILENAME = "abcdefgh_1.mzML"

def classify(classes, feature):
    for class_ in classes:
        if feature >= class_[0] and feature <= class_[1]:
            return str(class_)

def generate_coordinates(data_size, number_of_rows, row_length, column_length):
    single_scan_size = (data_size + 1) / number_of_rows
    column_size = math.floor(single_scan_size \
        * (column_length / (row_length + column_length)))
    row_size = single_scan_size - column_size 
    coordinates = []

    x_coord = 0
    y_coord = 0
    row_scan = True
    scanning_right = True
    for _ in range(0, number_of_rows):
        for i in range(1, single_scan_size + 1):
            coordinates.append((x_coord, y_coord))
            # Switching the horizontal scanning to the vertical scanning.
            if i % row_size == 0:
                row_scan = not row_scan
            if row_scan:
                if scanning_right:
                    x_coord += 1
                else:
                    x_coord -= 1
            else:
                y_coord += 1
        # Reversing scanner's direction.
        row_scan = not row_scan
        scanning_right = not scanning_right
    return coordinates


def main():
    run = pymzml.run.Reader(DATA_PATH + DATA_FILENAME)

    # Iterating through the data and storing into a temporary array (`Pymzml Reader' has no length attribute)
    preprocessed_mnis = []  
    for spectrum in run:
        for mass, intensity in spectrum.peaks:
            if intensity > INTENSITY_FLOOR:
                preprocessed_mnis.append((mass, intensity, spectrum['id']))
    mnis_dtype = [('mass', float), ('intensity', float), ('id', int)]

    # Using numpy to sort the data.
    mnis = np.array(preprocessed_mnis, dtype=mnis_dtype)
    mnis.sort(order='mass') 
    
    # Defining mass ranges for words.
    words = []
    starting_class_mass = mnis[0][0]
    previous_mass = mnis[0][0]
    for entity in mnis:
        mass = entity[0]
        tolerance = (previous_mass / 1000000) * TOLERANCE_FACTOR
        if mass - previous_mass > tolerance:
            words.append((starting_class_mass, previous_mass))
            starting_class_mass = mass
        previous_mass = mass 
    words.append((starting_class_mass, previous_mass))

    # Generating coordinates.
    number_of_rows = 8
    number_of_docs = 6327
    row_length = 62
    column_length = 1.25
    coordinates = generate_coordinates(
        number_of_docs, number_of_rows, row_length, column_length)

    # Populating the corpus.
    corpus = {}
    for entity in mnis:
        print(entity)
        key = str(entity[2] - 1)
        if key not in corpus:
            corpus[key] = {}
        class_ = classify(words, entity[0])
        if class_ not in corpus[key]:
            corpus[key][class_] = 0
        corpus[key][class_] += entity[1]

    corpus_series = pd.Series(corpus)
    corpus_series.to_pickle('../heavy_pickles/corpus.pickle')


if __name__ == '__main__':
    main()
